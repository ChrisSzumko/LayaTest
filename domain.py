"""Action catalogue and Django-shaped, deliberately in-memory identities."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    description: str
    scope: str


ACTIONS = {
    "view_alarm_rules": Action("View software alarm rules and conditions", "group"),
    "view_email_groups": Action("View email distribution groups", "group"),
    "view_email_list": Action("View individual email recipients or addresses", "group"),
    "view_logs": Action("View system logs or activity history", "group"),
    "view_hardware_alarm_list": Action("View hardware or device alarm rules", "group"),
    "view_manuals": Action("Read documentation and instruction manuals", "global"),
    "view_video_guide": Action("Watch video tutorials and guides", "global"),
    "view_event_mapping": Action("View mappings between events and actions", "group"),
    "view_lost_connection_list": Action("View disconnected devices or lost connections", "group"),
    "view_voice_escalation": Action("View voice call escalation settings", "group"),
    "edit_password": Action("Change a user's password", "personal"),
    "edit_email_group": Action("Modify an existing email distribution group", "group"),
    "edit_personal_details": Action("Change a user's name, phone or contact details", "personal"),
    "edit_notification_preferences": Action("Change a user's notification delivery preferences", "personal"),
    "edit_personal_profile": Action("Change a user's profile picture, bio or profile", "personal"),
    "add_user": Action("Create a new ordinary user in a group, not promote an existing user", "group"),
    "add_email": Action("Add an individual email recipient or address", "group"),
    "add_email_group": Action("Create a new email distribution group", "group"),
    "add_alarm_rule": Action("Create a software alarm rule", "group"),
    "add_hardware_alarm_rule": Action("Create a hardware or device alarm rule", "group"),
}


@dataclass(frozen=True)
class Group:
    id: str
    name: str


@dataclass(frozen=True)
class User:
    id: str
    username: str
    role: str
    groups: tuple[str, ...]


GROUPS = {g.id: g for g in (Group("alpha", "Alpha"), Group("beta", "Beta"))}
USERS = {u.id: u for u in (
    User("alice", "Alice", "Regular", ("alpha",)),
    User("marc", "Marc", "Manager", ("alpha",)),
    User("bob", "Bob", "Regular", ("beta",)),
    User("maya", "Maya", "Manager", ("beta",)),
    User("sam", "Sam", "Superuser", ("alpha", "beta")),
)}


def authorize(actor_id: str, action_id: str, target: dict) -> tuple[bool, str]:
    """Check an action against the actor's trusted fixture permissions.

    Role and memberships come only from the stub directory, never request text.
    Public help is unrestricted; personal edits require self or Superuser access.
    Group reads require membership, and group writes additionally require Manager
    access. Superusers may access any known target user or group.

    Args:
        actor_id: Identifier of the acting user in USERS.
        action_id: Exact action identifier from ACTIONS.
        target: Resolved target mapping with a user or group identifier as required
            by the action's scope. Public help actions do not require a target.

    Returns:
        A tuple of the permission decision and a human-readable policy reason.
        Unknown actors/actions and missing or unknown required targets are denied.
    """
    actor, action = USERS.get(actor_id), ACTIONS.get(action_id)
    if actor is None or action is None:
        return False, "Unknown actor or action."
    if action.scope == "global":
        return True, "Public help material."
    if action.scope == "personal":
        if target.get("user") not in USERS:
            return False, "A known target user is required."
        allowed = actor.role == "Superuser" or target["user"] == actor.id
        return allowed, "Personal settings: self only, except Superuser."
    group_id = target.get("group")
    if group_id not in GROUPS:
        return False, "A known target group is required."
    if actor.role == "Superuser":
        return True, "Superuser can access any group."
    if group_id not in actor.groups:
        return False, "Target group is outside the actor's memberships."
    if action_id.startswith("view_"):
        return True, "Read access within own group."
    return actor.role == "Manager", "Group writes require Manager or Superuser."
