import json

import pytest
from fastapi.testclient import TestClient

from app.engine import (
    ValidationError,
    _compare_selected_columns,
    prune,
    review,
)
from app.main import app
from app.models import PruneRequest, ReviewRequest
from pydantic import ValidationError as PydanticValidationError


def call_prune(payload):
    return prune(PruneRequest.model_validate(payload))


def call_review(payload):
    return review(ReviewRequest.model_validate(payload))


def ids(nodes):
    return [n["id"] for n in nodes]


def by_id(nodes):
    return {n["id"]: n for n in nodes}


# A graph with an unrelated branch (other_src -> other) the export never uses,
# an internal wide projection and a sensitive source column.
GRAPH = {
    "nodes": [
        {
            "id": "users",
            "type": "source",
            "columns": [
                {"name": "uid", "sensitive": False},
                {"name": "email", "sensitive": True},
                {"name": "country", "sensitive": False},
            ],
        },
        {
            "id": "only_cn",
            "type": "filter",
            "input": "users",
            "references": ["country"],
        },
        {
            "id": "export_users",
            "type": "projection",
            "input": "only_cn",
            "columns": [
                {"name": "user_id", "copy": "uid"},
                {"name": "mail", "copy": "email"},
                {"name": "bucket", "constant": "ALL"},
            ],
        },
        {
            "id": "other_src",
            "type": "source",
            "columns": [{"name": "z", "sensitive": True}],
        },
        {
            "id": "other",
            "type": "projection",
            "input": "other_src",
            "columns": [{"name": "z", "copy": "z"}],
        },
    ],
    "export": "export_users",
}


# ----------------------------- happy paths --------------------------------


def test_selecting_all_columns_drops_unreachable_branch_only():
    res = call_prune({**GRAPH, "columns": ["user_id", "mail", "bucket"]})
    # No trailing projection: the export already exposes exactly these.
    assert res["export"] == "export_users"
    assert res["added_projection"] is None
    assert ids(res["nodes"]) == ["users", "only_cn", "export_users"]
    assert res["dropped_nodes"] == ["other_src", "other"]
    assert res["pruned"] is True
    assert res["columns"] == ["user_id", "mail", "bucket"]
    # Reachable originals are returned byte-for-byte.
    assert res["nodes"] == GRAPH["nodes"][:3]


def test_subset_trims_export_projection_in_place_preserving_order():
    res = call_prune({**GRAPH, "columns": ["bucket", "user_id"]})
    # The export projection is trimmed in place: no trailing projection.
    assert res["export"] == "export_users"
    assert res["added_projection"] is None
    assert res["dropped_nodes"] == ["other_src", "other"]

    export_node = by_id(res["nodes"])["export_users"]
    # Unselected 'mail' is gone; result order follows the export node, not the
    # request order; the constant survives untouched.
    assert export_node["columns"] == [
        {"name": "user_id", "copy": "uid"},
        {"name": "bucket", "constant": "ALL"},
    ]
    assert res["columns"] == ["user_id", "bucket"]
    # The now-useless copy dependency 'email' is not demanded anywhere; the
    # source stays intact (source columns are never trimmed).
    assert res["nodes"][0] == GRAPH["nodes"][0]
    assert ids(res["nodes"]) == ["users", "only_cn", "export_users"]

    cols = {c["name"]: c for c in res["review"]["columns"]}
    assert set(cols) == {"user_id", "bucket"}
    assert [c["name"] for c in res["review"]["columns"]] == ["user_id", "bucket"]


def test_filter_export_still_gets_trailing_projection_for_unselected_pass_through():
    # A filter export passes every surviving input column through; when its
    # predicate keeps an unselected column alive, a trailing projection selects.
    payload = {
        "nodes": [
            {"id": "s", "type": "source", "columns": [
                {"name": "a"}, {"name": "b"}, {"name": "c"}]},
            {"id": "f", "type": "filter", "input": "s", "references": ["b"]},
        ],
        "export": "f",
        "columns": ["a", "c"],
    }
    res = call_prune(payload)
    assert res["export"] == "f__export"
    assert res["added_projection"] == "f__export"
    # The filter keeps its condition and passes a, b, c through; the trailing
    # projection narrows the final export to the selection (in original order).
    assert by_id(res["nodes"])["f"]["references"] == ["b"]
    trailing = res["nodes"][-1]
    assert trailing == {
        "id": "f__export",
        "type": "projection",
        "input": "f",
        "columns": [
            {"name": "a", "copy": "a"},
            {"name": "c", "copy": "c"},
        ],
    }
    assert res["columns"] == ["a", "c"]


def test_source_export_with_subset_gets_trailing_projection():
    # Source nodes are never trimmed, so a source-export subset needs the
    # trailing projection even though the demand pass finds nothing to trim.
    payload = {
        "nodes": [
            {"id": "s", "type": "source",
             "columns": [{"name": "a"}, {"name": "b", "sensitive": True}]},
        ],
        "export": "s",
        "columns": ["a"],
    }
    res = call_prune(payload)
    assert res["export"] == "s__export"
    assert [c["name"] for c in res["nodes"][0]["columns"]] == ["a", "b"]
    assert res["nodes"][-1]["columns"] == [{"name": "a", "copy": "a"}]


def test_selected_content_and_row_reasons_match_original():
    # 'mail' carries sensitive content; reasons must survive pruning unchanged.
    res = call_prune({**GRAPH, "columns": ["mail"]})
    col = res["review"]["columns"][0]
    assert col["name"] == "mail"
    assert col["content_reasons"] == [{"node_id": "users", "column": "email"}]
    assert col["row_reasons"] == []
    assert col["blocked"] is True
    # Pruning does not sanitize: the chosen column stays blocked.
    assert res["review"]["blocked"] is True


def test_row_influence_is_preserved_through_pruning():
    payload = {
        "nodes": [
            {"id": "s", "type": "source",
             "columns": [{"name": "id_card", "sensitive": True},
                         {"name": "score", "sensitive": False},
                         {"name": "region", "sensitive": False}]},
            {"id": "f", "type": "filter", "input": "s",
             "references": ["id_card", "region"]},
            {"id": "e", "type": "projection", "input": "f",
             "columns": [{"name": "score", "copy": "score"},
                         {"name": "region", "copy": "region"}]},
        ],
        "export": "e",
        "columns": ["score"],
    }
    res = call_prune(payload)
    # Filter stays intact (still references id_card), so row taint survives.
    filt = by_id(res["nodes"])["f"]
    assert filt["references"] == ["id_card", "region"]
    col = res["review"]["columns"][0]
    assert col["row_reasons"] == [{"node_id": "s", "column": "id_card"}]
    assert col["blocked"] is True


def test_pruned_graph_round_trips_through_review():
    res = call_prune({**GRAPH, "columns": ["bucket", "user_id"]})
    re_reviewed = call_review({"nodes": res["nodes"], "export": res["export"]})
    assert re_reviewed == res["review"]
    assert [c["name"] for c in re_reviewed["columns"]] == ["user_id", "bucket"]


def test_source_columns_are_never_trimmed():
    payload = {
        "nodes": [
            {"id": "s", "type": "source", "columns": [
                {"name": "a"}, {"name": "b"}, {"name": "c", "sensitive": True}]},
            {"id": "p", "type": "projection", "input": "s",
             "columns": [{"name": "a", "copy": "a"}, {"name": "c", "copy": "c"}]},
        ],
        "export": "p",
        "columns": ["a"],
    }
    res = call_prune(payload)
    source = by_id(res["nodes"])["s"]
    assert [c["name"] for c in source["columns"]] == ["a", "b", "c"]
    # Sensitive flag is preserved untouched.
    assert source["columns"][2] == {"name": "c", "sensitive": True}


def test_internal_projection_is_trimmed_to_demanded_outputs():
    payload = {
        "nodes": [
            {"id": "s", "type": "source", "columns": [
                {"name": "k"}, {"name": "v"}, {"name": "drop"}]},
            {"id": "wide", "type": "projection", "input": "s", "columns": [
                {"name": "k", "copy": "k"},
                {"name": "v", "copy": "v"},
                {"name": "d", "copy": "drop"},
                {"name": "lit", "constant": 1}]},
            {"id": "e", "type": "projection", "input": "wide",
             "columns": [{"name": "v", "copy": "v"}]},
        ],
        "export": "e",
        "columns": ["v"],
    }
    res = call_prune(payload)
    wide = by_id(res["nodes"])["wide"]
    # Only 'v' is demanded; the constant and the dropped copies are removed,
    # original relative order of the survivors is retained.
    assert [c["name"] for c in wide["columns"]] == ["v"]
    # The trailing selection projection is unnecessary (e already emits just v).
    assert res["export"] == "e" and res["added_projection"] is None


def test_constant_only_projection_keeps_its_input_relation():
    # The internal projection copies nothing, yet its input relation must stay.
    payload = {
        "nodes": [
            {"id": "src", "type": "source", "columns": [{"name": "a"}, {"name": "b"}]},
            {"id": "con", "type": "projection", "input": "src",
             "columns": [{"name": "x", "copy": "a"}, {"name": "y", "constant": 1}]},
            {"id": "e", "type": "projection", "input": "con",
             "columns": [{"name": "z", "constant": "Q"}]},
        ],
        "export": "e",
        "columns": ["z"],
    }
    res = call_prune(payload)
    assert ids(res["nodes"]) == ["src", "con", "e"]
    # con still emits at least its first output (the relation stays non-empty)
    # even though nothing it produces reaches the selected constant column.
    assert by_id(res["nodes"])["con"]["columns"][0]["name"] == "x"


def test_join_keeps_both_keys_and_both_side_relations():
    payload = {
        "nodes": [
            {"id": "s", "type": "source", "columns": [
                {"name": "k", "sensitive": True}, {"name": "v"}]},
            {"id": "r", "type": "source", "columns": [{"name": "k"}, {"name": "info"}]},
            {"id": "j", "type": "join",
             "left_input": "s", "right_input": "r",
             "left_key": "k", "right_key": "k",
             "outputs": [
                 {"alias": "v", "side": "left", "column": "v"},
                 {"alias": "info", "side": "right", "column": "info"}]},
        ],
        "export": "j",
        "columns": ["v"],
    }
    res = call_prune(payload)
    j = by_id(res["nodes"])["j"]
    # The export join is trimmed in place to the selected output...
    assert [o["alias"] for o in j["outputs"]] == ["v"]
    # ...keys are retained on both sides, and both side relations stay.
    assert j["left_key"] == "k" and j["right_key"] == "k"
    assert ids(res["nodes"]) == ["s", "r", "j"]
    assert res["export"] == "j" and res["added_projection"] is None
    col = res["review"]["columns"][0]
    # Sensitive left key taints rows even though only 'v' is exported.
    assert col["row_reasons"] == [{"node_id": "s", "column": "k"}]


def test_internal_join_trimmed_to_demanded_outputs_but_keys_always_kept():
    payload = {
        "nodes": [
            {"id": "s", "type": "source", "columns": [
                {"name": "k"}, {"name": "v"}, {"name": "x"}]},
            {"id": "r", "type": "source", "columns": [{"name": "k"}, {"name": "info"}]},
            {"id": "j", "type": "join",
             "left_input": "s", "right_input": "r",
             "left_key": "k", "right_key": "k",
             "outputs": [
                 {"alias": "v", "side": "left", "column": "v"},
                 {"alias": "x", "side": "left", "column": "x"},
                 {"alias": "info", "side": "right", "column": "info"}]},
            {"id": "e", "type": "projection", "input": "j",
             "columns": [{"name": "v", "copy": "v"}]},
        ],
        "export": "e",
        "columns": ["v"],
    }
    res = call_prune(payload)
    j = by_id(res["nodes"])["j"]
    # Only the demanded 'v' alias survives; keys on both sides are always kept.
    assert [o["alias"] for o in j["outputs"]] == ["v"]
    assert j["left_key"] == "k" and j["right_key"] == "k"
    # Both side relations remain (the right relation exists for the key alone).
    assert ids(res["nodes"]) == ["s", "r", "j", "e"]


def test_internal_join_trimmed_to_demanded_aliases_with_first_output_fallback():
    payload = {
        "nodes": [
            {"id": "l", "type": "source",
             "columns": [{"name": "k"}, {"name": "v1"}, {"name": "v2"}]},
            {"id": "r", "type": "source", "columns": [{"name": "k"}, {"name": "w"}]},
            {"id": "j", "type": "join",
             "left_input": "l", "right_input": "r",
             "left_key": "k", "right_key": "k",
             "outputs": [
                 {"alias": "o1", "side": "left", "column": "v1"},
                 {"alias": "o2", "side": "left", "column": "v2"}]},
            {"id": "constp", "type": "projection", "input": "j",
             "columns": [{"name": "lit", "constant": 9}]},
            {"id": "e", "type": "projection", "input": "constp",
             "columns": [{"name": "lit", "copy": "lit"}]},
        ],
        "export": "e",
        "columns": ["lit"],
    }
    res = call_prune(payload)
    j = by_id(res["nodes"])["j"]
    # j is retained (its relation feeds constp) but no alias is demanded:
    # keep the first output and both keys, never emit an empty output list.
    assert [o["alias"] for o in j["outputs"]] == ["o1"]
    assert j["left_key"] == "k" and j["right_key"] == "k"
    # e already exposes exactly the selected 'lit', so no trailing projection.
    assert ids(res["nodes"]) == ["l", "r", "j", "constp", "e"]
    assert res["export"] == "e" and res["added_projection"] is None


def test_shared_node_union_demands_from_all_retained_branches():
    payload = {
        "nodes": [
            {"id": "s", "type": "source", "columns": [
                {"name": "a"}, {"name": "b"}, {"name": "k"},
                {"name": "secret", "sensitive": True}]},
            {"id": "p", "type": "projection", "input": "s", "columns": [
                {"name": "a", "copy": "a"},
                {"name": "b", "copy": "b"},
                {"name": "k", "copy": "k"},
                {"name": "sec", "copy": "secret"}]},
            {"id": "r", "type": "source", "columns": [{"name": "k"}, {"name": "x"}]},
            {"id": "f", "type": "filter", "input": "p", "references": ["sec"]},
            {"id": "j", "type": "join",
             "left_input": "f", "right_input": "r",
             "left_key": "k", "right_key": "k",
             "outputs": [
                 {"alias": "a", "side": "left", "column": "a"},
                 {"alias": "b", "side": "left", "column": "b"},
                 {"alias": "x", "side": "right", "column": "x"}]},
        ],
        "export": "j",
        "columns": ["a", "x"],
    }
    res = call_prune(payload)
    p = by_id(res["nodes"])["p"]
    # The export join is trimmed in place to a,x; it needs a,k (left key) and
    # x,k (right key); the retained filter additionally needs sec. Only the
    # truly unused copy 'b' is gone, and shared-branch demands are unioned.
    assert [c["name"] for c in p["columns"]] == ["a", "k", "sec"]
    # The export join itself is trimmed to the selection; no trailing node.
    j = by_id(res["nodes"])["j"]
    assert [o["alias"] for o in j["outputs"]] == ["a", "x"]
    assert res["export"] == "j" and res["added_projection"] is None
    col = {c["name"]: c for c in res["review"]["columns"]}
    # Filter on sensitive sec taints every row of the join result.
    assert col["a"]["row_reasons"] == [{"node_id": "s", "column": "secret"}]
    assert col["x"]["row_reasons"] == [{"node_id": "s", "column": "secret"}]


def test_self_join_retains_shared_source_and_key():
    payload = {
        "nodes": [
            {"id": "s", "type": "source",
             "columns": [{"name": "k"}, {"name": "v1"}, {"name": "v2"}]},
            {"id": "j", "type": "join",
             "left_input": "s", "right_input": "s",
             "left_key": "k", "right_key": "k",
             "outputs": [
                 {"alias": "a", "side": "left", "column": "v1"},
                 {"alias": "b", "side": "right", "column": "v2"}]},
        ],
        "export": "j",
        "columns": ["b"],
    }
    res = call_prune(payload)
    # The single shared source is not duplicated; the export join is trimmed
    # in place so no trailing projection is needed.
    assert ids(res["nodes"]) == ["s", "j"]


def test_selected_join_column_demands_its_side_and_both_keys():
    # Export join selects only a right-side output: the left relation still
    # has to survive (left key shapes the row set), and its unused output
    # columns do not.
    payload = {
        "nodes": [
            {"id": "l", "type": "source", "columns": [
                {"name": "k"}, {"name": "v1"}, {"name": "v2"}]},
            {"id": "r", "type": "source", "columns": [
                {"name": "k"}, {"name": "w1"}, {"name": "w2"}]},
            {"id": "j", "type": "join",
             "left_input": "l", "right_input": "r",
             "left_key": "k", "right_key": "k",
             "outputs": [
                 {"alias": "a", "side": "left", "column": "v1"},
                 {"alias": "w", "side": "right", "column": "w1"},
                 {"alias": "z", "side": "right", "column": "w2"}]},
        ],
        "export": "j",
        "columns": ["w"],
    }
    res = call_prune(payload)
    j = by_id(res["nodes"])["j"]
    assert j["outputs"] == [{"alias": "w", "side": "right", "column": "w1"}]
    assert j["left_key"] == "k" and j["right_key"] == "k"
    # Both source relations survive (sources never trimmed anyway).
    assert ids(res["nodes"]) == ["l", "r", "j"]
    assert res["export"] == "j" and res["added_projection"] is None


def test_trimming_export_projection_drops_only_its_unused_copy_chain():
    # Unselected export copies must stop demanding their upstream chain; a
    # selected constant and a selected copy behave independently.
    payload = {
        "nodes": [
            {"id": "s", "type": "source", "columns": [
                {"name": "keep", "sensitive": True}, {"name": "drop"}]},
            {"id": "wide", "type": "projection", "input": "s", "columns": [
                {"name": "k", "copy": "keep"},
                {"name": "d", "copy": "drop"},
                {"name": "lit", "constant": 7}]},
            {"id": "e", "type": "projection", "input": "wide", "columns": [
                {"name": "k", "copy": "k"},
                {"name": "d", "copy": "d"},
                {"name": "lit", "copy": "lit"}]},
        ],
        "export": "e",
        "columns": ["k", "lit"],
    }
    res = call_prune(payload)
    nodes = by_id(res["nodes"])
    assert [c["name"] for c in nodes["e"]["columns"]] == ["k", "lit"]
    # 'wide' keeps only what the trimmed export reads: k plus the (internal)
    # constant lit; the d -> drop copy chain is gone.
    assert [c["name"] for c in nodes["wide"]["columns"]] == ["k", "lit"]
    assert nodes["wide"]["columns"][1] == {"name": "lit", "constant": 7}
    # Sensitive provenance of 'k' is preserved end to end.
    col = res["review"]["columns"][0]
    assert col["name"] == "k"
    assert col["content_reasons"] == [{"node_id": "s", "column": "keep"}]
    assert res["export"] == "e" and res["added_projection"] is None


def test_filter_reference_column_not_selected_is_still_review_identical():
    # Sensitive predicate column not among the selected columns must still
    # taint every selected row: filter condition and input column survive.
    payload = {
        "nodes": [
            {"id": "s", "type": "source", "columns": [
                {"name": "id_card", "sensitive": True},
                {"name": "score", "sensitive": False}]},
            {"id": "e", "type": "filter", "input": "s",
             "references": ["id_card"]},
        ],
        "export": "e",
        "columns": ["score"],
    }
    res = call_prune(payload)
    assert by_id(res["nodes"])["e"]["references"] == ["id_card"]
    # The filter still passes id_card through, so a trailing projection picks
    # score alone...
    assert res["export"] == "e__export"
    col = res["review"]["columns"][0]
    assert col["name"] == "score"
    assert col["row_reasons"] == [{"node_id": "s", "column": "id_card"}]


def test_node_and_column_relative_order_is_preserved():
    # Nodes given out of order; output must keep the original relative order.
    payload = {
        "nodes": [
            {"id": "p", "type": "projection", "input": "s",
             "columns": [{"name": "z", "copy": "b"}, {"name": "y", "copy": "a"}]},
            {"id": "s", "type": "source",
             "columns": [{"name": "a"}, {"name": "b"}, {"name": "c"}]},
        ],
        "export": "p",
        "columns": ["y", "z"],
    }
    res = call_prune(payload)
    # Original list order is preserved (p stays before s), original IDs kept.
    assert [n["id"] for n in res["nodes"]][:2] == ["p", "s"]
    # Column order follows the export node (z then y), not the request order.
    assert res["columns"] == ["z", "y"]
    assert [c["name"] for c in res["review"]["columns"]] == ["z", "y"]


def test_constants_and_conditions_are_not_modified():
    payload = {
        "nodes": [
            {"id": "s", "type": "source", "columns": [{"name": "a"}]},
            {"id": "f", "type": "filter", "input": "s", "references": ["a"]},
            {"id": "p", "type": "projection", "input": "f", "columns": [
                {"name": "n", "constant": None},
                {"name": "x", "constant": "OBJ"},
                {"name": "a", "copy": "a"}]},
        ],
        "export": "p",
        "columns": ["n", "x"],
    }
    res = call_prune(payload)
    p = by_id(res["nodes"])["p"]
    # The export projection is trimmed in place; surviving constants — an
    # explicit null included — are returned byte-for-byte (no spurious copy).
    assert p["columns"] == [
        {"name": "n", "constant": None},
        {"name": "x", "constant": "OBJ"},
    ]
    assert res["export"] == "p" and res["added_projection"] is None
    # The constant-only export still demands its input relation; the filter
    # condition is untouched.
    assert by_id(res["nodes"])["f"]["references"] == ["a"]
    assert ids(res["nodes"]) == ["s", "f", "p"]


def test_trailing_projection_id_avoids_collision():
    # A filter export passes the predicate column through, so a trailing
    # projection is genuinely required. An ancestor already occupies the
    # generated id; the new node must be renamed deterministically.
    payload = {
        "nodes": [
            {"id": "s", "type": "source",
             "columns": [{"name": "a"}, {"name": "b"}]},
            {"id": "f__export", "type": "projection", "input": "s",
             "columns": [{"name": "a", "copy": "a"}, {"name": "b", "copy": "b"}]},
            {"id": "f", "type": "filter", "input": "f__export",
             "references": ["b"]},
        ],
        "export": "f",
        "columns": ["a"],
    }
    res = call_prune(payload)
    assert res["export"] == "f__export_2"
    assert res["added_projection"] == "f__export_2"
    assert {n["id"] for n in res["nodes"]} == {
        "s", "f__export", "f", "f__export_2"
    }
    trailing = res["nodes"][-1]
    assert trailing["input"] == "f"
    assert trailing["columns"] == [{"name": "a", "copy": "a"}]


def test_pruned_graph_is_plain_json_and_resubmittable():
    res = call_prune({**GRAPH, "columns": ["mail"]})
    encoded = json.dumps({"nodes": res["nodes"], "export": res["export"]})
    decoded = json.loads(encoded)
    reviewed = call_review(decoded)
    assert [c["name"] for c in reviewed["columns"]] == ["mail"]
    assert reviewed["columns"][0]["content_reasons"] == [
        {"node_id": "users", "column": "email"}
    ]


# ----------------------------- rejection cases -----------------------------


def expect_engine_error(payload, code):
    with pytest.raises(ValidationError) as exc:
        call_prune(payload)
    assert exc.value.code == code
    return exc.value


def test_unknown_selected_column_rejected():
    exc = expect_engine_error({**GRAPH, "columns": ["user_id", "ghost"]},
                              "PRUNE_COLUMN_NOT_FOUND")
    assert exc.node_id == "export_users"
    assert exc.detail()["columns"] == ["ghost"]


def test_unknown_selected_column_rejected_even_if_graph_otherwise_safe():
    payload = {
        "nodes": [{"id": "s", "type": "source", "columns": [{"name": "a"}]}],
        "export": "s",
        "columns": ["nope"],
    }
    expect_engine_error(payload, "PRUNE_COLUMN_NOT_FOUND")


def test_in_place_column_shrink_at_node_limit_succeeds_without_append():
    # Regression: a 60-node graph whose export projection can shrink in place
    # must succeed. The old implementation left the export intact and tried to
    # append a 61st node, falsely reporting PRUNE_NODE_LIMIT.
    nodes = [{"id": "s0", "type": "source",
              "columns": [{"name": "a"}, {"name": "b"}]}]
    for i in range(1, 60):
        nodes.append({
            "id": f"n{i}",
            "type": "projection",
            "input": f"n{i - 1}" if i > 1 else "s0",
            "columns": [{"name": "a", "copy": "a"}, {"name": "b", "copy": "b"}],
        })
    res = call_prune({"nodes": nodes, "export": "n59", "columns": ["a"]})
    assert len(res["nodes"]) == 60
    assert res["export"] == "n59"
    assert res["added_projection"] is None
    # Every retained projection is narrowed to 'a', export included.
    assert all(
        [c["name"] for c in n["columns"]] == ["a"]
        for n in res["nodes"] if n["type"] == "projection"
    )
    assert [c["name"] for c in res["review"]["columns"]] == ["a"]
    # The 60-node result stays re-submittable to /api/review as-is.
    re_reviewed = call_review({"nodes": res["nodes"], "export": res["export"]})
    assert re_reviewed == res["review"]


def test_genuinely_needed_61st_node_at_limit_rejected_without_partial_graph():
    # A filter export keeps its predicate column in the pass-through relation,
    # so narrowing the columns genuinely requires the trailing projection:
    # the 61st node is unavoidable and the request is rejected.
    nodes = [{"id": "s0", "type": "source",
              "columns": [{"name": "a"}, {"name": "b"}]}]
    for i in range(1, 59):
        nodes.append({
            "id": f"n{i}",
            "type": "projection",
            "input": f"n{i - 1}" if i > 1 else "s0",
            "columns": [{"name": "a", "copy": "a"}, {"name": "b", "copy": "b"}],
        })
    nodes.append({
        "id": "f59", "type": "filter", "input": "n58", "references": ["b"],
    })
    exc = expect_engine_error(
        {"nodes": nodes, "export": "f59", "columns": ["a"]},
        "PRUNE_NODE_LIMIT",
    )
    detail = exc.detail()
    assert detail["node_count"] == 61 and detail["limit"] == 60
    assert detail["node_id"] == "f59"


def test_selecting_all_at_node_limit_still_succeeds_without_append():
    nodes = [{"id": "s0", "type": "source",
              "columns": [{"name": "a"}, {"name": "b"}]}]
    for i in range(1, 60):
        nodes.append({
            "id": f"n{i}",
            "type": "projection",
            "input": f"n{i - 1}" if i > 1 else "s0",
            "columns": [{"name": "a", "copy": "a"}, {"name": "b", "copy": "b"}],
        })
    res = call_prune({"nodes": nodes, "export": "n59", "columns": ["a", "b"]})
    assert len(res["nodes"]) == 60
    assert res["export"] == "n59"
    assert res["added_projection"] is None


def test_original_graph_validation_still_applies():
    # Invalid reference in a branch the export/prune would otherwise drop.
    expect_engine_error(
        {
            "nodes": [
                {"id": "s", "type": "source", "columns": [{"name": "a"}]},
                {"id": "ok", "type": "projection", "input": "s",
                 "columns": [{"name": "a", "copy": "a"}]},
                {"id": "bad", "type": "filter", "input": "s",
                 "references": ["ghost"]},
            ],
            "export": "ok",
            "columns": ["a"],
        },
        "INVALID_REFERENCE",
    )
    # Unknown export node.
    expect_engine_error(
        {
            "nodes": [{"id": "s", "type": "source", "columns": [{"name": "a"}]}],
            "export": "missing",
            "columns": ["a"],
        },
        "INVALID_REFERENCE",
    )
    # Dependency cycle.
    expect_engine_error(
        {
            "nodes": [
                {"id": "x", "type": "projection", "input": "y",
                 "columns": [{"name": "a", "constant": 1}]},
                {"id": "y", "type": "projection", "input": "x",
                 "columns": [{"name": "a", "constant": 1}]},
            ],
            "export": "x",
            "columns": ["a"],
        },
        "CYCLE",
    )
    # Duplicate node id.
    expect_engine_error(
        {
            "nodes": [
                {"id": "s", "type": "source", "columns": [{"name": "a"}]},
                {"id": "s", "type": "source", "columns": [{"name": "b"}]},
            ],
            "export": "s",
            "columns": ["a"],
        },
        "DUPLICATE_NODE",
    )


def test_review_mismatch_helper_detects_every_kind_of_difference():
    def col(name, content, rows, blocked=None):
        return {"name": name,
                "blocked": bool(content or rows) if blocked is None else blocked,
                "content_reasons": content, "row_reasons": rows}

    origin = [{"node_id": "s", "column": "secret"}]
    original = {"columns": [col("a", origin, []), col("b", [], [])]}
    # 'a' weakened content, 'b' missing, unexpected 'c' appended.
    pruned = {"columns": [col("a", [], []), col("c", [], [])]}
    diffs = _compare_selected_columns(original, pruned, ["a", "b"])
    assert diffs["missing_columns"] == ["b"]
    assert diffs["extra_columns"] == ["c"]
    assert diffs["actual_order"] == ["a", "c"]
    assert diffs["mismatched_columns"][0]["name"] == "a"
    assert diffs["mismatched_columns"][0]["pruned"]["content_reasons"] == []

    # An exact match yields no differences.
    exact = {"columns": [col("a", origin, []), col("b", [], [])]}
    assert _compare_selected_columns(original, exact, ["a", "b"]) == {}


# ----------------------------- request validation -------------------------


def test_empty_columns_list_rejected():
    with pytest.raises(PydanticValidationError):
        PruneRequest.model_validate({
            "nodes": [{"id": "s", "type": "source", "columns": [{"name": "a"}]}],
            "export": "s",
            "columns": [],
        })


def test_missing_columns_field_rejected():
    with pytest.raises(PydanticValidationError):
        PruneRequest.model_validate({
            "nodes": [{"id": "s", "type": "source", "columns": [{"name": "a"}]}],
            "export": "s",
        })


def test_duplicate_selected_columns_rejected():
    with pytest.raises(PydanticValidationError):
        PruneRequest.model_validate({
            "nodes": [{"id": "s", "type": "source", "columns": [{"name": "a"}]}],
            "export": "s",
            "columns": ["a", "a"],
        })


def test_blank_selected_column_rejected():
    with pytest.raises(PydanticValidationError):
        PruneRequest.model_validate({
            "nodes": [{"id": "s", "type": "source", "columns": [{"name": "a"}]}],
            "export": "s",
            "columns": [""],
        })


def test_prune_unknown_field_rejected():
    with pytest.raises(PydanticValidationError):
        PruneRequest.model_validate({
            "nodes": [{"id": "s", "type": "source", "columns": [{"name": "a"}]}],
            "export": "s",
            "columns": ["a"],
            "nope": 1,
        })


def test_more_than_60_nodes_rejected_at_schema():
    nodes = [{"id": "s0", "type": "source", "columns": [{"name": "a"}]}]
    for i in range(1, 61):
        nodes.append({"id": f"n{i}", "type": "projection",
                      "input": f"n{i - 1}" if i > 1 else "s0",
                      "columns": [{"name": "a", "copy": "a"}]})
    with pytest.raises(PydanticValidationError):
        PruneRequest.model_validate({"nodes": nodes, "export": "n60",
                                     "columns": ["a"]})


# ----------------------------- HTTP endpoint ------------------------------


client = TestClient(app)


def test_http_prune_success_and_review_follow_up():
    resp = client.post("/api/prune", json={**GRAPH, "columns": ["bucket", "user_id"]})
    assert resp.status_code == 200
    body = resp.json()
    # The export projection is trimmed in place; no trailing projection.
    assert body["export"] == "export_users"
    assert body["added_projection"] is None
    assert [c["name"] for c in body["review"]["columns"]] == ["user_id", "bucket"]

    follow = client.post("/api/review",
                         json={"nodes": body["nodes"], "export": body["export"]})
    assert follow.status_code == 200
    follow_body = follow.json()
    assert [c["name"] for c in follow_body["columns"]] == ["user_id", "bucket"]
    assert follow_body["blocked"] is False


def test_http_prune_unknown_column_returns_400_without_partial_graph():
    resp = client.post("/api/prune", json={**GRAPH, "columns": ["ghost"]})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "PRUNE_COLUMN_NOT_FOUND"
    assert "nodes" not in body and "review" not in body


def test_http_prune_node_limit_returns_400():
    # Filter export keeps the referenced column alive, so the selection
    # genuinely needs a 61st node.
    nodes = [{"id": "s0", "type": "source",
              "columns": [{"name": "a"}, {"name": "b"}]}]
    for i in range(1, 59):
        nodes.append({"id": f"n{i}", "type": "projection",
                      "input": f"n{i - 1}" if i > 1 else "s0",
                      "columns": [{"name": "a", "copy": "a"},
                                  {"name": "b", "copy": "b"}]})
    nodes.append({"id": "f59", "type": "filter", "input": "n58",
                  "references": ["b"]})
    resp = client.post("/api/prune",
                       json={"nodes": nodes, "export": "f59", "columns": ["a"]})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "PRUNE_NODE_LIMIT"


def test_http_prune_in_place_shrink_at_node_limit_succeeds():
    nodes = [{"id": "s0", "type": "source",
              "columns": [{"name": "a"}, {"name": "b"}]}]
    for i in range(1, 60):
        nodes.append({"id": f"n{i}", "type": "projection",
                      "input": f"n{i - 1}" if i > 1 else "s0",
                      "columns": [{"name": "a", "copy": "a"},
                                  {"name": "b", "copy": "b"}]})
    resp = client.post("/api/prune",
                       json={"nodes": nodes, "export": "n59", "columns": ["a"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["export"] == "n59"
    assert body["added_projection"] is None
    assert len(body["nodes"]) == 60
    assert [c["name"] for c in body["review"]["columns"]] == ["a"]


def test_http_prune_schema_error_returns_422():
    resp = client.post("/api/prune", json={
        "nodes": [{"id": "s", "type": "source", "columns": [{"name": "a"}]}],
        "export": "s",
        "columns": ["a", "a"],
    })
    assert resp.status_code == 422


def test_http_prune_original_graph_error_returns_400():
    resp = client.post("/api/prune", json={
        "nodes": [{"id": "s", "type": "source", "columns": [{"name": "a"}]}],
        "export": "missing",
        "columns": ["a"],
    })
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "INVALID_REFERENCE"


def test_existing_endpoints_behavior_unchanged():
    review_resp = client.post("/api/review", json=GRAPH)
    assert review_resp.status_code == 200
    body = review_resp.json()
    assert body["export"] == "export_users"
    # Review never carries the prune-specific selection/status fields.
    assert "pruned" not in body and "dropped_nodes" not in body
    assert "added_projection" not in body

    revise_resp = client.post("/api/revise", json={
        **{k: GRAPH[k] for k in ("nodes", "export")},
        "allowed_replacements": ["mail"],
    })
    assert revise_resp.status_code == 200
    assert revise_resp.json()["export"] == "export_users__safe_export"
