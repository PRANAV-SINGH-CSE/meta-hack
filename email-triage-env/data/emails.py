"""
Synthetic email corpus with ground-truth annotations.

Each record has:
  email        — the Email object
  ground_truth — dict of correct answers for graders
"""

from typing import Any, Dict, List
from env.models import Email

# ─────────────────────────────────────────────────────────────────────────────
#  TASK 1  —  Spam vs Not-Spam  (10 emails, easy)
# ─────────────────────────────────────────────────────────────────────────────

TASK1_EMAILS: List[Dict[str, Any]] = [
    {
        "email": Email(
            id="t1_001",
            subject="URGENT: You've won $5,000,000!!!",
            body=(
                "Congratulations! Your email was selected in our annual lottery. "
                "Click http://win-now.biz/claim to claim your prize TODAY. "
                "Limited time offer. Act now or lose your winnings FOREVER."
            ),
            sender="Prize Department",
            sender_email="prizes@win-now.biz",
            timestamp="2024-03-01T09:00:00Z",
            thread_length=1,
        ),
        "ground_truth": {"spam_label": "spam"},
    },
    {
        "email": Email(
            id="t1_002",
            subject="Q2 roadmap review — your input needed",
            body=(
                "Hi team,\n\nWe're finalising the Q2 product roadmap and would love "
                "your feedback on the attached draft. Please review and reply with "
                "comments by Friday EOD.\n\nThanks,\nSarah"
            ),
            sender="Sarah Mitchell",
            sender_email="sarah.mitchell@company.com",
            timestamp="2024-03-01T09:15:00Z",
            thread_length=1,
        ),
        "ground_truth": {"spam_label": "not_spam"},
    },
    {
        "email": Email(
            id="t1_003",
            subject="Make $10K/month from home — NO experience needed",
            body=(
                "Are you tired of your 9-to-5? Our proven system lets anyone earn "
                "$10,000 per month working just 2 hours a day. Reply YES to get "
                "started. Unsubscribe if you don't want financial freedom."
            ),
            sender="Wealth Secrets",
            sender_email="info@wealth-secrets247.net",
            timestamp="2024-03-01T10:00:00Z",
            thread_length=1,
        ),
        "ground_truth": {"spam_label": "spam"},
    },
    {
        "email": Email(
            id="t1_004",
            subject="Invoice #4821 — due 15 March",
            body=(
                "Dear Accounts Team,\n\nPlease find attached Invoice #4821 for "
                "consulting services rendered in February 2024. Total: $3,200.00. "
                "Payment is due by 15 March 2024.\n\nBest regards,\nAlex Turner"
            ),
            sender="Alex Turner",
            sender_email="alex@turnerdesigns.co",
            timestamp="2024-03-01T11:00:00Z",
            thread_length=1,
        ),
        "ground_truth": {"spam_label": "not_spam"},
    },
    {
        "email": Email(
            id="t1_005",
            subject="Your account has been COMPROMISED — verify now",
            body=(
                "We detected suspicious activity on your account. Your account will "
                "be suspended in 24 hours unless you verify your identity at "
                "http://secure-login.accounts-verify.com. Enter your password "
                "and credit card to restore access."
            ),
            sender="Security Team",
            sender_email="security@accounts-verify.com",
            timestamp="2024-03-01T11:30:00Z",
            thread_length=1,
        ),
        "ground_truth": {"spam_label": "spam"},
    },
    {
        "email": Email(
            id="t1_006",
            subject="Team lunch this Thursday?",
            body=(
                "Hey everyone,\n\nThinking of organising a team lunch this Thursday "
                "at 12:30 at The Green Fork. Let me know if you can make it!\n\nCheers,\nPriya"
            ),
            sender="Priya Nair",
            sender_email="priya.nair@company.com",
            timestamp="2024-03-01T12:00:00Z",
            thread_length=1,
        ),
        "ground_truth": {"spam_label": "not_spam"},
    },
    {
        "email": Email(
            id="t1_007",
            subject="Exclusive VIP offer — Rolex watches 90% off",
            body=(
                "As a VIP customer you qualify for our EXCLUSIVE sale. "
                "Brand-new Rolex watches from $199. Limited stock. "
                "Order now: http://luxury-deals-shop.ru/rolex"
            ),
            sender="Luxury Deals",
            sender_email="deals@luxury-deals-shop.ru",
            timestamp="2024-03-01T13:00:00Z",
            thread_length=1,
        ),
        "ground_truth": {"spam_label": "spam"},
    },
    {
        "email": Email(
            id="t1_008",
            subject="Re: Onboarding checklist for new hire — James Chen",
            body=(
                "Hi HR team,\n\nJust confirming that James Chen starts Monday. "
                "Could you send over the equipment checklist and IT setup form "
                "so we can get everything ready?\n\nThanks,\nRodrigo"
            ),
            sender="Rodrigo Santos",
            sender_email="rodrigo.santos@company.com",
            timestamp="2024-03-01T14:00:00Z",
            thread_length=2,
        ),
        "ground_truth": {"spam_label": "not_spam"},
    },
    {
        "email": Email(
            id="t1_009",
            subject="Nigerian Prince needs your help URGENTLY",
            body=(
                "DEAR FRIEND, I am Prince Emmanuel of Nigeria. I have $48 million "
                "USD that I cannot access due to government restrictions. I need "
                "a trusted foreign partner. You will receive 30% for your help. "
                "Please reply with your bank details URGENTLY."
            ),
            sender="Prince Emmanuel",
            sender_email="prince.emmanuel1@yahoo.ru",
            timestamp="2024-03-01T15:00:00Z",
            thread_length=1,
        ),
        "ground_truth": {"spam_label": "spam"},
    },
    {
        "email": Email(
            id="t1_010",
            subject="Contract renewal — please review by end of week",
            body=(
                "Hi,\n\nAttached is the updated service contract for renewal. "
                "Key changes are in sections 3.2 (SLA terms) and 7 (pricing). "
                "Please review and let us know if you have questions before "
                "signing.\n\nKind regards,\nLinda Park, Legal"
            ),
            sender="Linda Park",
            sender_email="linda.park@legalpartners.com",
            timestamp="2024-03-01T16:00:00Z",
            thread_length=1,
        ),
        "ground_truth": {"spam_label": "not_spam"},
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  TASK 2  —  Priority + Department Routing  (10 emails, medium)
# ─────────────────────────────────────────────────────────────────────────────

TASK2_EMAILS: List[Dict[str, Any]] = [
    {
        "email": Email(
            id="t2_001",
            subject="Production database DOWN — all services affected",
            body=(
                "CRITICAL: The production PostgreSQL cluster went down 10 minutes ago. "
                "All customer-facing services are returning 500 errors. "
                "The on-call engineer is investigating but needs immediate backup. "
                "Revenue impact is accumulating. Please escalate NOW."
            ),
            sender="Ops Alert System",
            sender_email="alerts@ops.company.com",
            timestamp="2024-03-04T03:15:00Z",
            thread_length=1,
        ),
        "ground_truth": {"priority": "urgent", "department": "technical"},
    },
    {
        "email": Email(
            id="t2_002",
            subject="Question about our refund policy",
            body=(
                "Hello,\n\nI purchased your Pro plan last month and would like to "
                "know if I'm eligible for a refund since I no longer need the service. "
                "My order number is #ORD-8821. There's no rush — I just want to "
                "understand my options.\n\nThank you!"
            ),
            sender="Marcus Webb",
            sender_email="m.webb@gmail.com",
            timestamp="2024-03-04T09:30:00Z",
            thread_length=1,
        ),
        "ground_truth": {"priority": "low", "department": "billing"},
    },
    {
        "email": Email(
            id="t2_003",
            subject="Enterprise deal — $250K opportunity needs demo ASAP",
            body=(
                "Hi,\n\nI'm the VP of Engineering at Nexus Corp. We're evaluating "
                "vendors for a company-wide deployment (1,200 seats). Budget is "
                "approved. We need a technical demo this week before our board meeting "
                "on Friday. Can someone from your enterprise team reach out today?"
            ),
            sender="David Kim",
            sender_email="d.kim@nexuscorp.com",
            timestamp="2024-03-04T08:00:00Z",
            thread_length=1,
        ),
        "ground_truth": {"priority": "urgent", "department": "sales"},
    },
    {
        "email": Email(
            id="t2_004",
            subject="Monthly newsletter — March edition",
            body=(
                "Hi there!\n\nHere's your March product newsletter. This month we "
                "shipped dark mode, improved our mobile app, and fixed 47 bugs. "
                "Read the full update on our blog. As always, reply with any "
                "feedback you have.\n\nThe Product Team"
            ),
            sender="Product Team",
            sender_email="newsletter@company.com",
            timestamp="2024-03-04T10:00:00Z",
            thread_length=1,
        ),
        "ground_truth": {"priority": "low", "department": "general"},
    },
    {
        "email": Email(
            id="t2_005",
            subject="Bug: data export corrupts CSV files — 3 customers affected",
            body=(
                "Hi support team,\n\nWe're seeing an issue where the CSV export "
                "feature produces corrupted files when the dataset has more than "
                "10,000 rows. Three enterprise customers have reported this today. "
                "It's blocking their monthly reporting. Needs a fix ASAP."
            ),
            sender="Jake Morrison",
            sender_email="jake.morrison@company.com",
            timestamp="2024-03-04T11:00:00Z",
            thread_length=3,
        ),
        "ground_truth": {"priority": "high", "department": "technical"},
    },
    {
        "email": Email(
            id="t2_006",
            subject="Expense report submission — Feb 2024",
            body=(
                "Hi Finance,\n\nPlease find attached my expense report for February 2024. "
                "Total: $847.50 (flights + hotel for the Chicago conference). "
                "Receipts are included. Please process when you get a chance.\n\nThanks,\nAnna"
            ),
            sender="Anna Schmidt",
            sender_email="anna.schmidt@company.com",
            timestamp="2024-03-04T12:00:00Z",
            thread_length=1,
        ),
        "ground_truth": {"priority": "low", "department": "hr"},
    },
    {
        "email": Email(
            id="t2_007",
            subject="Payment failed — subscription about to expire tomorrow",
            body=(
                "Hello,\n\nMy credit card payment failed and my subscription expires "
                "tomorrow. I've updated my card details but the charge hasn't gone "
                "through yet. I'm in the middle of a critical project and cannot "
                "afford downtime. Please process manually if possible."
            ),
            sender="Carla Reyes",
            sender_email="c.reyes@designstudio.io",
            timestamp="2024-03-04T14:00:00Z",
            thread_length=2,
        ),
        "ground_truth": {"priority": "high", "department": "billing"},
    },
    {
        "email": Email(
            id="t2_008",
            subject="Partnership proposal — co-marketing opportunity",
            body=(
                "Hi,\n\nWe run a complementary SaaS product with 50,000 users and "
                "believe there's a strong co-marketing opportunity. We'd love to "
                "explore a joint webinar or blog swap. Happy to schedule a call "
                "next week if you're interested."
            ),
            sender="Tom Bradley",
            sender_email="tom@partnerhq.com",
            timestamp="2024-03-04T15:00:00Z",
            thread_length=1,
        ),
        "ground_truth": {"priority": "medium", "department": "sales"},
    },
    {
        "email": Email(
            id="t2_009",
            subject="Harassment complaint — immediate HR attention required",
            body=(
                "To HR,\n\nI need to formally report an incident of workplace "
                "harassment that occurred during yesterday's team meeting. "
                "I would like this treated as confidential and addressed urgently. "
                "I am available to speak with an HR rep at any time today."
            ),
            sender="[Confidential]",
            sender_email="anon.report@company.com",
            timestamp="2024-03-04T08:45:00Z",
            thread_length=1,
        ),
        "ground_truth": {"priority": "urgent", "department": "hr"},
    },
    {
        "email": Email(
            id="t2_010",
            subject="Feature request: dark mode for mobile app",
            body=(
                "Hey,\n\nLove the product! One thing I'd love to see is dark mode "
                "on the mobile app — the desktop version has it but mobile doesn't. "
                "Not urgent at all, just wanted to log it as a suggestion. Keep up "
                "the great work!\n\nCheers,\nNadia"
            ),
            sender="Nadia Flores",
            sender_email="nadia.flores@example.com",
            timestamp="2024-03-04T16:30:00Z",
            thread_length=1,
        ),
        "ground_truth": {"priority": "low", "department": "support"},
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  TASK 3  —  Full Triage  (5 emails, hard)
# ─────────────────────────────────────────────────────────────────────────────

TASK3_EMAILS: List[Dict[str, Any]] = [
    {
        "email": Email(
            id="t3_001",
            subject="Server outage — 3 enterprise clients offline",
            body=(
                "Hi,\n\nWe have a CRITICAL situation. Three of our largest enterprise "
                "clients (Nexus Corp, Atlas Financial, and Meridian Health) are all "
                "reporting that the API gateway is returning 503 errors. This started "
                "approximately 20 minutes ago. SLA breach will occur in 40 minutes. "
                "We've already had two escalation calls and clients are threatening "
                "to invoke penalty clauses. The on-call engineer needs authorisation "
                "to perform a full gateway restart. Who can approve?"
            ),
            sender="Ops Lead",
            sender_email="ops.lead@company.com",
            timestamp="2024-03-05T02:00:00Z",
            thread_length=4,
        ),
        "ground_truth": {
            "priority": "urgent",
            "department": "technical",
            "required_action_items": [
                "approve gateway restart",
                "notify enterprise clients",
                "escalate to CTO/VP Engineering",
                "monitor SLA timer",
            ],
            "required_reply_keywords": ["approve", "escalat", "contact", "sla"],
        },
    },
    {
        "email": Email(
            id="t3_002",
            subject="Annual performance review scheduling — all managers",
            body=(
                "Dear Managers,\n\nAs we approach the end of Q1, it's time to "
                "schedule annual performance reviews for all direct reports. "
                "Reviews must be completed and submitted to HR by April 30th. "
                "Please use the HR portal to book 1:1 sessions. Ensure you have "
                "reviewed each employee's self-assessment before the meeting. "
                "Training on the new performance framework is available in the "
                "LMS — completion is mandatory before conducting reviews. "
                "Contact hr@company.com with any questions."
            ),
            sender="HR Department",
            sender_email="hr@company.com",
            timestamp="2024-03-05T09:00:00Z",
            thread_length=1,
        ),
        "ground_truth": {
            "priority": "medium",
            "department": "hr",
            "required_action_items": [
                "complete lms training",
                "schedule review meetings",
                "review self-assessments",
                "submit by april 30",
            ],
            "required_reply_keywords": ["acknowledge", "schedule", "complete", "deadline"],
        },
    },
    {
        "email": Email(
            id="t3_003",
            subject="Chargeback dispute — $12,400 at risk",
            body=(
                "Hello Billing Team,\n\nWe have received a chargeback notification "
                "from our payment processor for $12,400 from customer account "
                "#ACC-7721 (Stratford Consulting). The customer claims they never "
                "authorised the charge, but our records show a signed contract and "
                "successful delivery of services. We have 7 calendar days to submit "
                "evidence to dispute the chargeback or the funds will be permanently "
                "reversed. Please gather: signed contract, delivery confirmation, "
                "and all invoice correspondence."
            ),
            sender="Payments Team",
            sender_email="payments@company.com",
            timestamp="2024-03-05T10:30:00Z",
            thread_length=2,
        ),
        "ground_truth": {
            "priority": "high",
            "department": "billing",
            "required_action_items": [
                "gather signed contract",
                "gather delivery confirmation",
                "gather invoice correspondence",
                "submit dispute within 7 days",
            ],
            "required_reply_keywords": ["dispute", "evidence", "deadline", "gather"],
        },
    },
    {
        "email": Email(
            id="t3_004",
            subject="Sales proposal review — Cascade Industries (250 seats)",
            body=(
                "Hi,\n\nI'm meeting with Cascade Industries on Thursday to present "
                "our enterprise proposal. They're a 250-seat opportunity, currently "
                "evaluating us vs Competitor X. Key concerns from their side: "
                "1) Data residency in EU, 2) SSO/SAML support, 3) Dedicated account "
                "manager. I need product team confirmation that all three are "
                "available on our Enterprise tier, and I need legal to review the "
                "DPA (data processing agreement) before Thursday. Can both teams "
                "confirm availability by Wednesday noon?"
            ),
            sender="James Okafor",
            sender_email="james.okafor@company.com",
            timestamp="2024-03-05T11:00:00Z",
            thread_length=1,
        ),
        "ground_truth": {
            "priority": "high",
            "department": "sales",
            "required_action_items": [
                "confirm eu data residency",
                "confirm sso/saml support",
                "assign dedicated account manager",
                "legal review dpa",
                "respond by wednesday noon",
            ],
            "required_reply_keywords": ["confirm", "legal", "wednesday", "enterprise"],
        },
    },
    {
        "email": Email(
            id="t3_005",
            subject="Customer data breach report — possible GDPR violation",
            body=(
                "To the Security and Legal teams,\n\nWe have identified a potential "
                "data breach affecting approximately 2,300 user records. A "
                "misconfigured S3 bucket was publicly accessible for an estimated "
                "72 hours between Feb 26–28. Exposed data may include: email "
                "addresses, hashed passwords, and billing addresses. Under GDPR "
                "Article 33, we are required to notify the supervisory authority "
                "within 72 hours of becoming aware. That deadline is in approximately "
                "18 hours. We also need to assess notification obligations to affected "
                "users under Article 34. Legal counsel required immediately."
            ),
            sender="InfoSec Team",
            sender_email="infosec@company.com",
            timestamp="2024-03-05T14:00:00Z",
            thread_length=3,
        ),
        "ground_truth": {
            "priority": "urgent",
            "department": "technical",
            "required_action_items": [
                "notify supervisory authority within 18 hours",
                "engage legal counsel",
                "assess article 34 user notification",
                "secure s3 bucket immediately",
                "document breach details",
            ],
            "required_reply_keywords": ["gdpr", "authority", "legal", "notify", "urgent"],
        },
    },
]


# ─── Index by task ─────────────────────────────────────────────────────────────

TASK_DATASETS = {
    "task1_spam": TASK1_EMAILS,
    "task2_routing": TASK2_EMAILS,
    "task3_full_triage": TASK3_EMAILS,
}
