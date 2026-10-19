"""Compile exact journal effects, state and receipt into one native command."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import re

from potpie_context_engine.core.graph_journal import JournalError, journal_json

from ._mutation_execution import _mutation_result_to_dict

_STATE = "PotpieRevision"
_RECEIPT = "PotpieMutationReceipt"
_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def publish_query(before, after, *, state, receipt, result=None, execution=None):
    params = {
        "pot": state.pot_id,
        "state": journal_json(state),
        "generation": state.generation,
        "old_head": receipt.parent_commit_id if receipt else state.head,
    }
    clauses = [f"MATCH (v:{_STATE} {{pot_id:$pot}}) WHERE v.journal=$state"]
    changed = [after[key] for key in after if before.get(key) != after[key]]
    for i, record in enumerate(sorted(changed, key=lambda r: r.kind != "entity")):
        props = deepcopy(dict(record.fields["properties"]))
        props["retired"] = not record.fields.get("active", True)
        props["record_id"] = record.record_id
        params[f"rid{i}"] = record.record_id
        params[f"props{i}"] = props
        if record.kind == "entity":
            labels = record.fields["labels"]
            old_labels = (
                before[record.record_id].fields["labels"]
                if record.record_id in before
                else ()
            )
            if any(not _LABEL.fullmatch(label) for label in (*labels, *old_labels)):
                raise JournalError("invalid native label")
            remove = sorted(set(old_labels) - set(labels))
            clauses.append(
                f"CALL {{ WITH v MERGE (e:Entity {{group_id:$pot,record_id:$rid{i}}}) "
                f"SET e=$props{i} "
                + (f"REMOVE e:{':'.join(remove)} " if remove else "")
                + (f"SET e:{':'.join(labels)} " if labels else "")
                + f"RETURN count(e) AS count{i} }}"
            )
        else:
            vector = props.pop("fact_embedding", None)
            if vector is not None:
                params[f"vector{i}"] = vector
            params.update(
                {
                    f"src{i}": record.fields["subject_record_id"],
                    f"dst{i}": record.fields["object_record_id"],
                }
            )
            clauses.append(
                f"CALL {{ WITH v MATCH (a:Entity {{group_id:$pot,record_id:$src{i}}}) "
                f"MATCH (b:Entity {{group_id:$pot,record_id:$dst{i}}}) "
                f"MERGE (a)-[r:RELATES_TO {{group_id:$pot,record_id:$rid{i}}}]->(b) SET r=$props{i} "
                + (
                    f"SET r.fact_embedding=vecf32($vector{i}) "
                    if vector is not None
                    else ""
                )
                + f"RETURN count(r) AS count{i} }}"
            )
    next_state = state
    if receipt:
        clauses[0] += (
            f" OPTIONAL MATCH (prior:{_RECEIPT} {{pot_id:$pot,mutation_id:$mid}}) "
            "WITH v,prior WHERE prior IS NULL"
        )
        next_state = replace(state, sequence=receipt.sequence, head=receipt.commit_id)
        params.update(
            mid=receipt.commit_id,
            fingerprint=receipt.fingerprint,
            journal=journal_json(receipt),
            payload=json.dumps(
                {
                    "mutation_summary": {
                        "entity_upserts_applied": result.entity_upserts_applied
                        if result
                        else 0,
                        "edge_upserts_applied": result.edge_upserts_applied
                        if result
                        else 0,
                        "edge_deletes_applied": result.edge_deletes_applied
                        if result
                        else 0,
                        "invalidations_applied": result.invalidations_applied
                        if result
                        else 0,
                        "stamp_counts": {},
                    }
                }
            ),
            entities=result.entity_upserts_applied if result else 0,
            edges=result.edge_upserts_applied if result else 0,
            deletes=result.edge_deletes_applied if result else 0,
            invalidations=result.invalidations_applied if result else 0,
            sequence=receipt.sequence,
        )
        if execution is not None:
            params["payload"] = json.dumps(
                _mutation_result_to_dict(execution), sort_keys=True
            )
            params["affected"] = list(execution.execution.affected_claim_record_ids)
            params["expected"] = execution.execution.version_before
            params["advance"] = (
                execution.execution.version_after - execution.execution.version_before
            )
            clauses[0] += " AND v.version=$expected"
        else:
            params["affected"], params["advance"] = [], 1
        clauses.append(
            f"CREATE (receipt:{_RECEIPT} {{pot_id:$pot,mutation_id:$mid,fingerprint:$fingerprint,"
            "payload:$payload,journal:$journal,journal_generation:$generation,sequence:$sequence,"
            "entities:$entities,edges:$edges,deletes:$deletes,invalidations:$invalidations,"
            "affected_claim_record_ids:$affected})"
        )
        clauses.append("SET v.version=v.version+$advance")
    params["next_state"] = journal_json(next_state)
    clauses.append("SET v.journal=$next_state RETURN v.version")
    return " ".join(clauses), params
