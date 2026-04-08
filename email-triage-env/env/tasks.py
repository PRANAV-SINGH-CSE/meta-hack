"""
Task registry — one entry per task.
"""

from env.models import TaskMeta

TASKS = {
    "task1_spam": TaskMeta(
        task_id="task1_spam",
        name="Spam vs. Not-Spam Classification",
        description=(
            "Classify each incoming email as 'spam' or 'not_spam'. "
            "Look for hallmarks of spam: suspicious senders, unrealistic promises, "
            "phishing links, and urgent pressure tactics. Legitimate emails tend "
            "to come from known domains and contain specific, contextual content. "
            "Required action field: spam_label."
        ),
        difficulty="easy",
        max_steps=10,
    ),
    "task2_routing": TaskMeta(
        task_id="task2_routing",
        name="Priority & Department Routing",
        description=(
            "Assign a priority level (low / medium / high / urgent) and route "
            "each email to the correct department "
            "(sales / support / billing / hr / technical / general). "
            "Consider urgency, business impact, time sensitivity, and topic. "
            "Required action fields: priority, department."
        ),
        difficulty="medium",
        max_steps=10,
    ),
    "task3_full_triage": TaskMeta(
        task_id="task3_full_triage",
        name="Full Email Triage",
        description=(
            "Perform complete triage on each email: "
            "(1) assign priority, "
            "(2) route to department, "
            "(3) list concrete action items the recipient must take, "
            "(4) draft a brief professional reply acknowledging the email and "
            "outlining next steps. "
            "Required action fields: priority, department, action_items, reply_draft."
        ),
        difficulty="hard",
        max_steps=5,
    ),
}
