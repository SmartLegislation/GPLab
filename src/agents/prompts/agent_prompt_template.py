import json
from typing import Dict, Any, List, Optional

def format_static_attributes(attributes: Dict[str, Any]) -> str:
    lines = ["## Your Static Attributes:"]
    for category, details in attributes.items():
        lines.append(f"### {category}")
        if isinstance(details, dict):
            for key, value in details.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append(f"- {category}: {details}") # Handle cases where top level is not a dict
    return "\n".join(lines)

def format_dynamic_cognition(memories: List[str], emotion: str) -> str:
    lines = ["## Your Current State:"]
    lines.append("### Recent Memories & Reflections (Most recent first):")
    if memories:
        for mem in memories:
            lines.append(f"- {mem}")
    else:
        lines.append("- No relevant memories found for the current situation.")
    lines.append(f"\n### Current Emotional State: {emotion}")
    return "\n".join(lines)

def format_environment_perception(perception: Dict[str, Any]) -> str:
    lines = ["## Current Environment:"]
    for system_name, info in perception.items():
        lines.append(f"### From {system_name}:")
        if isinstance(info, dict):
            for key, value in info.items():
                if key == "recommended_posts" and isinstance(value, list): # Special formatting for posts
                    lines.append("- Recommended Posts:")
                    if value:
                        for post in value[:5]: # Show limited posts
                            post_id = post.get('id') if post.get('id') else post.get('post_id','N/A')
                            lines.append(f"  - Post ID: {post_id}, Content: \"{post.get('content', '')}\" (Likes: {post.get('likes', 0)})")
                    else:
                        lines.append("  - No posts recommended.")
                elif isinstance(value, list): # Handle other list values
                    lines.append(f"- {key}:")
                    for item in value:
                        lines.append(f"  - {item}")
                elif isinstance(value, dict): # Handle nested dicts generically
                    lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False)}")
                else:
                    lines.append(f"- {key}: {value}")
        elif isinstance(info, list): # Handle if top-level perception is a list
             lines.append(f"- {system_name} Info:")
             for item in info:
                 lines.append(f"  - {item}")
        else:
            lines.append(f"- {system_name}: {info}")
            
    return "\n".join(lines)

def format_behavior_decision_schema(active_subsystems: Dict[str, Any], agent_decision_principles: List[str], example_decision: Optional[Dict] = None) -> str:
    lines = [
        "## Your Task:",
        "Based on your attributes, current state (memories, emotion), and the environment, decide on your actions for the current time step.",
        "You MUST respond ONLY with a single JSON object. The JSON object should have keys corresponding to the active social subsystems you interact with.",
        "For each subsystem, provide the requested decision attributes.",
        "Include a 'reasoning' field explaining *why* you made these decisions based on the provided context.",
        ""
    ]
    # Add principles from config
    if agent_decision_principles:
        lines.append("Your decisions must follow these principles:")
        for principle in agent_decision_principles:
            lines.append(f"- {principle}")

    lines.append("\n### Active Subsystems and Required Decisions:")
    for name, sys_instance in active_subsystems.items():
        # sys_instance.decision_attributes is List[Dict[str,str]]
        attr_names = [attr.get("name") for attr in sys_instance.decision_attributes if attr.get("name")]
        lines.append(f"- **{name}**: Requires decisions for: `{', '.join(attr_names)}`")

    lines.append("\n### Response Format (JSON ONLY):")
    lines.append("```json")
    # Construct a dynamic example based on active subsystems
    if example_decision:
         lines.append(json.dumps(example_decision, indent=2, ensure_ascii=False))
    else: # Generate a generic placeholder example
        example = {
            "reasoning": "Provide a brief justification here."
        }
        for name, sys_instance in active_subsystems.items():
            example[name] = {}
            for attr_config in sys_instance.decision_attributes:
                attr_name = attr_config.get("name")
                attr_desc = attr_config.get("description", "<your decision value>")
                if attr_name:
                     example[name][attr_name] = attr_desc
        lines.append(json.dumps(example, indent=2, ensure_ascii=False))
    lines.append("```")
    return "\n".join(lines)

def construct_agent_prompt(
    static_attributes: Dict[str, Any],
    relevant_memories: List[str],
    emotion: str,
    environment_perception: Dict[str, Any],
    active_subsystems: Dict[str, Any], # Dict of {name: instance}
    agent_decision_principles: List[str],
    example_decision: Optional[Dict] = None
) -> str:
    
    prompt_parts = [
        format_static_attributes(static_attributes),
        format_dynamic_cognition(relevant_memories, emotion),
        format_environment_perception(environment_perception),
        format_behavior_decision_schema(active_subsystems, agent_decision_principles, example_decision)
    ]
    
    return "\n\n".join(prompt_parts)





