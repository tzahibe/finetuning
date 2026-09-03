import json

from src.datasets.schema import RelationshipType, RoomType, ZoneType

SYSTEM_PROMPT = """You are an architectural planning assistant. Given a residential \
Brief (what the user wants), a Site (the physical plot), and a set of Constraints \
(facts the design must satisfy), output an Architectural SPEC as a single JSON object.

The SPEC must have exactly these top-level keys: "program", "zones", "relationships", "circulation".

- "program": list of {{"type": ROOM_TYPE, "count": int > 0, "area_per_room_m2": float > 0, "zone": ZONE_TYPE}}
  one entry per distinct room type in the design. "area_per_room_m2" is the area of
  ONE room of that type (not a total).
- "zones": list of {{"type": ZONE_TYPE, "room_types": [ROOM_TYPE, ...]}} grouping room
  types by zone.
- "relationships": list of {{"a_type": ROOM_TYPE, "b_type": ROOM_TYPE, "relationship": RELATIONSHIP_TYPE}}
  describing functional connections between room types.
- "circulation": list of ROOM_TYPE values that serve circulation (e.g. staircases).

Valid ROOM_TYPE values: {room_types}
Valid ZONE_TYPE values: {zone_types}
Valid RELATIONSHIP_TYPE values: {relationship_types}

Respond with ONLY the JSON object. No markdown code fences, no explanation, no \
reasoning before or after it.""".format(
    room_types=", ".join(t.value for t in RoomType),
    zone_types=", ".join(t.value for t in ZoneType),
    relationship_types=", ".join(t.value for t in RelationshipType),
)


def build_prompt(input_dict: dict) -> list[dict]:
    """Builds the chat-format prompt (system + user) for one baseline example.
    input_dict is the {"brief":..., "site":..., "constraints":...} dict from the
    SFT-serialized dataset (src/datasets/sft_format.py::to_sft_pair()'s "input").
    """
    user_content = "Brief:\n" + json.dumps(input_dict["brief"], indent=2)
    user_content += "\n\nSite:\n" + json.dumps(input_dict["site"], indent=2)
    user_content += "\n\nConstraints:\n" + json.dumps(input_dict["constraints"], indent=2)
    user_content += "\n\nReturn the Architectural SPEC as a single JSON object."

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
