"""
create_eval_pairs.py

Generates data/eval_pairs.json with 150+ labeled query pairs:
~75 true paraphrases (should_match: True) and ~75 hard negatives (should_match: False).
Specifically crafted for sentence-transformers (all-MiniLM-L6-v2) semantic similarity evaluation.
"""

import json
import os

EVAL_PAIRS = [
    # ==================== TRUE PARAPHRASES (76 pairs) ====================
    {"query_a": "How do I reset my password?", "query_b": "How can I reset my forgotten password?", "should_match": True},
    {"query_a": "What is your refund policy?", "query_b": "What is your policy regarding refunds?", "should_match": True},
    {"query_a": "What are your business hours?", "query_b": "What are your open customer service hours?", "should_match": True},
    {"query_a": "Is there a free trial available?", "query_b": "Do you offer a free trial period?", "should_match": True},
    {"query_a": "How do I contact customer support?", "query_b": "How can I contact your support team?", "should_match": True},
    {"query_a": "How do I track my package shipment?", "query_b": "How do I track my order shipment?", "should_match": True},
    {"query_a": "Can I pay with credit card?", "query_b": "Do you accept credit card payments?", "should_match": True},
    {"query_a": "How do I change my profile picture?", "query_b": "How do I update my profile avatar picture?", "should_match": True},
    {"query_a": "How do I export my account data?", "query_b": "How do I download an export of my account data?", "should_match": True},
    {"query_a": "What is the pricing for team plan?", "query_b": "How much does the team plan cost?", "should_match": True},
    {"query_a": "How do I invite team members?", "query_b": "How can I invite new members to my team?", "should_match": True},
    {"query_a": "Do you have an iOS app?", "query_b": "Is an iOS iPhone application available?", "should_match": True},
    {"query_a": "How do I change my account email address?", "query_b": "How can I update my primary email address?", "should_match": True},
    {"query_a": "What is the maximum file upload size?", "query_b": "What is the limit for file upload size?", "should_match": True},
    {"query_a": "How do I turn off email notifications?", "query_b": "How do I disable email notification alerts?", "should_match": True},
    {"query_a": "Can I use this software offline?", "query_b": "Does this software work without internet offline?", "should_match": True},
    {"query_a": "How do I delete my workspace?", "query_b": "How do I remove my workspace permanently?", "should_match": True},
    {"query_a": "Where can I download my invoice?", "query_b": "How do I download my payment invoice PDF?", "should_match": True},
    {"query_a": "How do I redeem a promo code?", "query_b": "Where do I apply a discount coupon code?", "should_match": True},
    {"query_a": "Do you offer student discounts?", "query_b": "Is there a discount available for university students?", "should_match": True},
    {"query_a": "How do I enable two-factor authentication?", "query_b": "How do I set up 2FA two-factor authentication?", "should_match": True},
    {"query_a": "What programming languages do you support?", "query_b": "Which programming languages are supported by your SDK?", "should_match": True},
    {"query_a": "How long does standard shipping take?", "query_b": "What is the estimated delivery time for standard shipping?", "should_match": True},
    {"query_a": "Can I transfer my software license?", "query_b": "How do I transfer my license to another machine?", "should_match": True},
    {"query_a": "How do I enable dark mode?", "query_b": "Where is the toggle to turn on dark mode theme?", "should_match": True},
    {"query_a": "What is your uptime SLA guarantee?", "query_b": "What service uptime SLA percentage do you guarantee?", "should_match": True},
    {"query_a": "How do I merge two accounts?", "query_b": "Is it possible to merge two user accounts together?", "should_match": True},
    {"query_a": "Where can I find API documentation?", "query_b": "Where is the developer API documentation located?", "should_match": True},
    {"query_a": "How do I change default language settings?", "query_b": "Where do I change the application display language?", "should_match": True},
    {"query_a": "Is my credit card information secure?", "query_b": "How secure is my stored credit card payment data?", "should_match": True},
    {"query_a": "How do I clear browser cache?", "query_b": "What are the steps to clear stored browser cache?", "should_match": True},
    {"query_a": "Do you support SAML single sign-on?", "query_b": "Can we configure SAML SSO single sign-on?", "should_match": True},
    {"query_a": "How do I generate an API secret key?", "query_b": "Where do I create a new API access key?", "should_match": True},
    {"query_a": "What happens when my free trial expires?", "query_b": "What occurs after my free trial period ends?", "should_match": True},
    {"query_a": "How do I submit a feature request?", "query_b": "Where can I submit new feature requests or feedback?", "should_match": True},
    {"query_a": "Can I customize the dashboard layout?", "query_b": "How do I customize the layout of my dashboard?", "should_match": True},
    {"query_a": "How do I revoke an API key?", "query_b": "How do I invalidate an existing API access key?", "should_match": True},
    {"query_a": "Which web browsers are supported?", "query_b": "What web browsers are compatible with your service?", "should_match": True},
    {"query_a": "How do I set up webhook notifications?", "query_b": "Where do I configure custom HTTP webhook alerts?", "should_match": True},
    {"query_a": "Do you offer non-profit organization discounts?", "query_b": "Is special discount pricing available for non-profits?", "should_match": True},
    {"query_a": "How do I report a security vulnerability?", "query_b": "Where do I report a potential security vulnerability?", "should_match": True},
    {"query_a": "How do I restore deleted files?", "query_b": "Can I restore recently deleted files from trash?", "should_match": True},
    {"query_a": "What is the API rate limit per minute?", "query_b": "How many requests per minute are permitted on the API?", "should_match": True},
    {"query_a": "How do I update my phone number?", "query_b": "Where can I update the phone number on my account?", "should_match": True},
    {"query_a": "Can I schedule automated daily backups?", "query_b": "How do I set up automated daily data backups?", "should_match": True},
    {"query_a": "How do I add a new payment method?", "query_b": "Where can I add a new credit card payment method?", "should_match": True},
    {"query_a": "What is the account cloud storage limit?", "query_b": "How much storage capacity is allocated to my account?", "should_match": True},
    {"query_a": "How do I filter table search results?", "query_b": "Where are the search filters to filter table results?", "should_match": True},
    {"query_a": "Can I share a file link publicly?", "query_b": "How do I create a public share link for a file?", "should_match": True},
    {"query_a": "How do I assign admin user roles?", "query_b": "Where do administrators set role permissions for users?", "should_match": True},
    {"query_a": "What is your privacy cookie policy?", "query_b": "Where can I read your official cookie privacy policy?", "should_match": True},
    {"query_a": "How do I view my order history?", "query_b": "Where can I see my past order purchase history?", "should_match": True},
    {"query_a": "Is there an Android mobile application?", "query_b": "Do you have an Android app available on Google Play?", "should_match": True},
    {"query_a": "How do I turn on push notifications?", "query_b": "Where do I enable push notifications in app settings?", "should_match": True},
    {"query_a": "How do I set account timezone?", "query_b": "Where do I configure my default account timezone?", "should_match": True},
    {"query_a": "Can I pause my monthly subscription?", "query_b": "Is it possible to temporarily pause my subscription?", "should_match": True},
    {"query_a": "How do I connect a custom domain?", "query_b": "Where do I map my custom DNS domain name?", "should_match": True},
    {"query_a": "What is customer support response time?", "query_b": "How fast does your support team respond to inquiries?", "should_match": True},
    {"query_a": "How do I archive an old project?", "query_b": "Where is the button to archive a completed project?", "should_match": True},
    {"query_a": "Can I integrate with Slack channels?", "query_b": "Does your platform offer an integration for Slack?", "should_match": True},
    {"query_a": "How do I configure email alerts?", "query_b": "Where do I set up automated email notification alerts?", "should_match": True},
    {"query_a": "What data import formats are supported?", "query_b": "Which file formats are supported for data import?", "should_match": True},
    {"query_a": "How do I switch to annual billing?", "query_b": "Where do I upgrade from monthly to yearly annual billing?", "should_match": True},
    {"query_a": "Do you support multi-factor auth MFA?", "query_b": "Can I require multi-factor authentication MFA for my team?", "should_match": True},
    {"query_a": "How do I leave a shared workspace?", "query_b": "Where is the option to leave a shared team workspace?", "should_match": True},
    {"query_a": "How do I check system uptime status?", "query_b": "Where is your live system status page located?", "should_match": True},
    {"query_a": "How do I update billing contact email?", "query_b": "Where do I change the billing recipient email address?", "should_match": True},
    {"query_a": "Can I export reports as CSV files?", "query_b": "How do I download dashboard report metrics as CSV?", "should_match": True},
    {"query_a": "How do I reset password without login?", "query_b": "Where is the forgot password recovery link?", "should_match": True},
    {"query_a": "What is the maximum team seat count?", "query_b": "How many user seats are included in the plan limit?", "should_match": True},
    {"query_a": "How do I enable auto-renewal?", "query_b": "Where do I turn on automatic subscription renewal?", "should_match": True},
    {"query_a": "Can I transfer project ownership?", "query_b": "How do I transfer ownership of a project to another admin?", "should_match": True},
    {"query_a": "How do I inspect application error logs?", "query_b": "Where can I view server error logs and stack traces?", "should_match": True},
    {"query_a": "What happens if credit card payment fails?", "query_b": "How many times will a failed payment charge be retried?", "should_match": True},
    {"query_a": "How do I configure API CORS settings?", "query_b": "Where do I set allowed origins for cross-origin CORS requests?", "should_match": True},
    {"query_a": "Can I customize email notification templates?", "query_b": "How do I edit the HTML design of outgoing email templates?", "should_match": True},

    # ==================== HARD NEGATIVES (76 pairs) ====================
    # 1. Action Reversals & Polarity (Cancel vs Pause, Upgrade vs Downgrade, Enable vs Disable)
    {"query_a": "How do I cancel my subscription?", "query_b": "How do I pause my subscription?", "should_match": False},
    {"query_a": "I want to cancel my account immediately.", "query_b": "I want to pause my account for 30 days.", "should_match": False},
    {"query_a": "How do I upgrade my plan to Pro?", "query_b": "How do I downgrade my plan to Free?", "should_match": False},
    {"query_a": "I need to upgrade my storage capacity limit.", "query_b": "I need to downgrade my storage capacity limit.", "should_match": False},
    {"query_a": "How do I enable two-factor authentication?", "query_b": "How do I disable two-factor authentication?", "should_match": False},
    {"query_a": "Turn on dark mode display theme.", "query_b": "Turn off dark mode display theme.", "should_match": False},
    {"query_a": "How to activate auto-renewal on subscription?", "query_b": "How to deactivate auto-renewal on subscription?", "should_match": False},
    {"query_a": "Enable public link sharing permissions.", "query_b": "Disable public link sharing permissions.", "should_match": False},
    {"query_a": "Grant admin access rights to team user.", "query_b": "Revoke admin access rights from team user.", "should_match": False},

    # 2. Entity / ID / Number mismatches
    {"query_a": "What is the current delivery status of order #1024?", "query_b": "What is the current delivery status of order #9981?", "should_match": False},
    {"query_a": "Show billing statement for account 5501.", "query_b": "Show billing statement for account 8820.", "should_match": False},
    {"query_a": "Transfer $500 payment to Alice Smith.", "query_b": "Transfer $500 payment to Bob Jones.", "should_match": False},
    {"query_a": "What is the weather forecast in Tokyo today?", "query_b": "What is the weather forecast in London today?", "should_match": False},
    {"query_a": "Inspect details for support ticket #4012.", "query_b": "Inspect details for support ticket #7731.", "should_match": False},
    {"query_a": "Delete database record for User A.", "query_b": "Delete database record for User B.", "should_match": False},
    {"query_a": "What are business hours for New York branch?", "query_b": "What are business hours for Sydney branch?", "should_match": False},
    {"query_a": "Show logs for server pod prod-us-east-1.", "query_b": "Show logs for server pod prod-eu-west-1.", "should_match": False},

    # 3. Shipping vs Billing vs Physical Address
    {"query_a": "Can I change my shipping address?", "query_b": "Can I change my billing address?", "should_match": False},
    {"query_a": "Update shipping destination for order #4001.", "query_b": "Update billing card address for order #4001.", "should_match": False},
    {"query_a": "Where is your corporate headquarters physical address?", "query_b": "Where do I send payment check billing address?", "should_match": False},

    # 4. Refund vs Exchange vs Store Credit
    {"query_a": "I want a full refund to my credit card.", "query_b": "I want to exchange my item for a different size.", "should_match": False},
    {"query_a": "Can I receive store credit for returned item?", "query_b": "Can I receive cash refund for returned item?", "should_match": False},

    # 5. Password Reset vs Username Recovery vs Account Deletion
    {"query_a": "How do I reset my account password?", "query_b": "How do I recover my forgotten username?", "should_match": False},
    {"query_a": "I forgot my password, how do I change it?", "query_b": "I want to delete my account permanently.", "should_match": False},

    # 6. Sign In vs Sign Up vs Sign Out
    {"query_a": "Where do I sign up for a new account?", "query_b": "Where do I sign in to my existing account?", "should_match": False},
    {"query_a": "How do I sign out of active session?", "query_b": "How do I sign in to new session?", "should_match": False},

    # 7. Import vs Export
    {"query_a": "How do I import contacts from a CSV file?", "query_b": "How do I export contacts to a CSV file?", "should_match": False},
    {"query_a": "Import SQL database backup dump.", "query_b": "Export SQL database backup dump.", "should_match": False},

    # 8. Public vs Private
    {"query_a": "How do I make my project repository public?", "query_b": "How do I make my project repository private?", "should_match": False},
    {"query_a": "Generate a public web access link.", "query_b": "Generate a private password protected link.", "should_match": False},

    # 9. Increase vs Decrease Limits
    {"query_a": "How do I increase my API rate limit cap?", "query_b": "How do I decrease my API rate limit cap?", "should_match": False},
    {"query_a": "Increase maximum thread worker count.", "query_b": "Decrease maximum thread worker count.", "should_match": False},

    # 10. Add vs Remove / Delete
    {"query_a": "How do I add a new credit card payment method?", "query_b": "How do I remove an existing credit card payment method?", "should_match": False},
    {"query_a": "How to add a new admin team user?", "query_b": "How to delete an admin team user?", "should_match": False},
    {"query_a": "Add custom domain DNS record.", "query_b": "Remove custom domain DNS record.", "should_match": False},

    # 11. Distinct Technical Stacks & Technologies
    {"query_a": "How to build a REST API in Python FastAPI?", "query_b": "How to build a REST API in Node.js Express?", "should_match": False},
    {"query_a": "Set LLM model temperature parameter to 0.0.", "query_b": "Set LLM model temperature parameter to 1.0.", "should_match": False},
    {"query_a": "Connect application backend to MySQL database.", "query_b": "Connect application backend to PostgreSQL database.", "should_match": False},
    {"query_a": "Deploy containerized app on Amazon Web Services EC2.", "query_b": "Deploy containerized app on Google Cloud Platform Run.", "should_match": False},
    {"query_a": "Run docker container service on port 8000.", "query_b": "Run docker container service on port 3000.", "should_match": False},
    {"query_a": "Install PyTorch library with CPU support.", "query_b": "Install PyTorch library with CUDA GPU support.", "should_match": False},
    {"query_a": "Configure NGINX web server reverse proxy.", "query_b": "Configure Apache HTTP web server proxy.", "should_match": False},

    # 12. Policy Inquiries vs Executing Immediate Destruction
    {"query_a": "What is the policy for deleting an account?", "query_b": "Delete my account permanently right now.", "should_match": False},
    {"query_a": "What are the rules for subscription cancellation?", "query_b": "Cancel my subscription immediately.", "should_match": False},

    # 13. Past vs Future Time Horizons
    {"query_a": "What were the revenue numbers for last year?", "query_b": "What are the projected revenue numbers for next year?", "should_match": False},
    {"query_a": "View server error logs from yesterday.", "query_b": "Schedule server maintenance logs for tomorrow.", "should_match": False},

    # 14. Additional Distinct Domain Questions
    {"query_a": "How to reset security PIN on debit card?", "query_b": "How to reset online banking password?", "should_match": False},
    {"query_a": "Download 64-bit Windows installation package.", "query_b": "Download macOS Apple Silicon installation package.", "should_match": False},
    {"query_a": "Check battery charge level on device.", "query_b": "Check remaining disk storage capacity on device.", "should_match": False},
    {"query_a": "Mute incoming notifications for 1 hour.", "query_b": "Mute incoming notifications permanently.", "should_match": False},
    {"query_a": "Search for invoices issued in 2024.", "query_b": "Search for invoices issued in 2026.", "should_match": False},
    {"query_a": "How to turn on Wi-Fi wireless networking?", "query_b": "How to turn on Bluetooth wireless pairing?", "should_match": False},
    {"query_a": "Set primary emergency contact person.", "query_b": "Set secondary emergency contact person.", "should_match": False},
    {"query_a": "What is current CPU utilization percentage?", "query_b": "What is current RAM memory utilization percentage?", "should_match": False},
    {"query_a": "Create a new git feature branch.", "query_b": "Delete existing git feature branch.", "should_match": False},
    {"query_a": "How to increase font size in code editor?", "query_b": "How to change font family in code editor?", "should_match": False},
    {"query_a": "Set up webhook integration for Slack.", "query_b": "Set up webhook integration for Discord.", "should_match": False},
    {"query_a": "How to restart proxy server process?", "query_b": "How to stop proxy server process?", "should_match": False},
    {"query_a": "What is the API rate limit for free plan?", "query_b": "What is the API rate limit for enterprise plan?", "should_match": False},
    {"query_a": "How to compress PNG image file size?", "query_b": "How to compress MP4 video file size?", "should_match": False},
    {"query_a": "Is my SSL certificate currently valid?", "query_b": "How to renew expired SSL certificate?", "should_match": False},
    {"query_a": "Clear local browser cookies data.", "query_b": "Clear browser search history log.", "should_match": False},
    {"query_a": "What is default network port for HTTP?", "query_b": "What is default network port for HTTPS?", "should_match": False},
    {"query_a": "Backup database data to AWS S3 bucket.", "query_b": "Backup database data to Azure Blob container.", "should_match": False},
    {"query_a": "How to create a new team workspace?", "query_b": "How to delete an existing team workspace?", "should_match": False},
    {"query_a": "Check status of flight flight #100.", "query_b": "Check status of flight flight #200.", "should_match": False},
    {"query_a": "Update credit card expiration date.", "query_b": "Update credit card CVV security code.", "should_match": False},
    {"query_a": "Set logging level to debug mode.", "query_b": "Set logging level to error mode.", "should_match": False},
    {"query_a": "View inbound HTTP network traffic.", "query_b": "View outbound HTTP network traffic.", "should_match": False},
    {"query_a": "Generate 2048-bit RSA encryption key.", "query_b": "Generate 4096-bit RSA encryption key.", "should_match": False},
    {"query_a": "What is outdoor temperature in Miami?", "query_b": "What is outdoor humidity in Miami?", "should_match": False},
    {"query_a": "Filter data table by creation date.", "query_b": "Sort data table by item price.", "should_match": False},
    {"query_a": "How to install Node.js version 18?", "query_b": "How to install Node.js version 20?", "should_match": False},
    {"query_a": "Show active user login sessions.", "query_b": "Show expired user login sessions.", "should_match": False},
    {"query_a": "How to mount network NFS storage volume?", "query_b": "How to unmount network NFS storage volume?", "should_match": False},
    {"query_a": "Check available disk space in GB.", "query_b": "Check available system RAM memory in GB.", "should_match": False},
    {"query_a": "Update DNS A record IP address.", "query_b": "Update DNS CNAME record domain.", "should_match": False},
    {"query_a": "Execute SQL SELECT query on database.", "query_b": "Execute SQL DELETE query on database.", "should_match": False},
]


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dirs = [
        os.path.join(base_dir, "..", "data"),
        os.path.join(base_dir, "data"),
    ]
    for tdir in target_dirs:
        os.makedirs(tdir, exist_ok=True)
        tpath = os.path.join(tdir, "eval_pairs.json")
        with open(tpath, "w", encoding="utf-8") as f:
            json.dump(EVAL_PAIRS, f, indent=2)

    trues = sum(1 for p in EVAL_PAIRS if p["should_match"])
    falses = sum(1 for p in EVAL_PAIRS if not p["should_match"])
    print(f"[DATASET] Successfully generated {len(EVAL_PAIRS)} pairs")
    print(f"  True Paraphrases: {trues}")
    print(f"  Hard Negatives:   {falses}")


if __name__ == "__main__":
    main()
