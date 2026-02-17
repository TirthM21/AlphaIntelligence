"""Email notification system for AlphaIntelligence Capital.

Sends institutional-grade HTML market intelligence briefings
via Gmail SMTP. Requires a Gmail account with App Password.
"""

import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Send institutional-grade emails for AlphaIntelligence Capital via Gmail SMTP."""
    
    def __init__(self):
        """Initialize email notifier with Gmail SMTP settings."""
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        
        # Get credentials from environment
        self.sender_email = os.getenv('EMAIL_SENDER')
        raw_password = os.getenv('EMAIL_PASSWORD', '')
        # Gmail App Passwords are displayed as 'xxxx xxxx xxxx xxxx'
        # but SMTP login requires the 16-char version without spaces
        self.sender_password = raw_password.strip().replace(' ', '') if raw_password else ''
        
        # Debug: log password length (NOT the password itself) for troubleshooting
        if self.sender_password:
            logger.debug(f"Email password loaded: {len(self.sender_password)} characters")
        self.recipient_email = os.getenv('EMAIL_RECIPIENT', self.sender_email)
        
        self.enabled = bool(self.sender_email and self.sender_password)
        
        if not self.enabled:
            logger.warning("Email notifications disabled - EMAIL_SENDER or EMAIL_PASSWORD not set")
        else:
            logger.info(f"Email notifications enabled - will send to {self.recipient_email}")
    
    def send_newsletter(
        self,
        newsletter_path: str,
        scan_report_path: Optional[str] = None,
        subject: Optional[str] = None
    ) -> bool:
        """Send daily newsletter via email.
        
        Args:
            newsletter_path: Path to newsletter markdown file
            scan_report_path: Optional path to full scan report (as attachment)
            subject: Email subject (auto-generated if None)
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.warning("Email not configured - skipping send")
            return False
        
        try:
            # Read newsletter content
            with open(newsletter_path, 'r', encoding='utf-8') as f:
                newsletter_md = f.read()
            
            # Prefer generated HTML companion, otherwise convert markdown with light template
            newsletter_html_path = Path(newsletter_path).with_suffix('.html')
            if newsletter_html_path.exists():
                newsletter_html = newsletter_html_path.read_text(encoding='utf-8')
            else:
                newsletter_html = self._markdown_to_html(newsletter_md)
            
            # Create subject
            if not subject:
                date_str = datetime.now().strftime('%B %d, %Y')
                subject = f"🏦 AlphaIntelligence Capital — Daily Brief | {date_str}"
            
            # Create email
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            
            # Add multipart body (plain text + HTML)
            plain_part = MIMEText(newsletter_md, 'plain', 'utf-8')
            html_part = MIMEText(newsletter_html, 'html', 'utf-8')
            msg.attach(plain_part)
            msg.attach(html_part)
            
            # Attach full scan report if provided
            if scan_report_path and Path(scan_report_path).exists():
                with open(scan_report_path, 'r', encoding='utf-8') as f:
                    attachment = MIMEText(f.read(), 'plain', 'utf-8')
                    attachment.add_header(
                        'Content-Disposition',
                        'attachment',
                        filename='full_scan_report.txt'
                    )
                    msg.attach(attachment)
            
            # Send email with debug logging
            logger.info(f"Connecting to {self.smtp_server}:{self.smtp_port}...")
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                logger.info(f"Authenticating as {self.sender_email}...")
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"✅ Newsletter sent to {self.recipient_email}")
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"❌ SMTP Authentication FAILED. Check EMAIL_SENDER and EMAIL_PASSWORD in .env")
            logger.error(f"   Gmail requires an App Password (not your regular password).")
            logger.error(f"   Error details: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to send newsletter: {type(e).__name__}: {e}")
            return False
    
    def send_screening_results(
        self,
        results,
        top_n: int = 10,
        subject: Optional[str] = None
    ) -> bool:
        """Send screening results via email.
        
        Args:
            results: DataFrame with screening results
            top_n: Number of top results to include
            subject: Email subject (auto-generated if None)
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.warning("Email not configured - skipping send")
            return False
        
        try:
            import pandas as pd
            
            # Get top results
            if isinstance(results, pd.DataFrame):
                top_results = results.head(top_n)
            else:
                top_results = results[:top_n]
            
            # Create markdown content
            date_str = datetime.now().strftime('%B %d, %Y')
            
            md_content = [f"# 📊 AlphaIntelligence Capital — Signal Report | {date_str}", ""]
            md_content.append(f"**Top {top_n} Buy Signals**")
            md_content.append("")
            md_content.append("| Ticker | Name | Sector | Price | Buy Signal | Value Score | Support Score | RSI |")
            md_content.append("|--------|------|--------|-------|------------|-------------|---------------|-----|")
            
            for _, row in top_results.iterrows():
                ticker = row.get('ticker', 'N/A')
                name = row.get('name', '')[:15]  # Truncate long names
                sector = row.get('sector', 'Unknown')[:10]
                price = row.get('current_price', 0)
                buy_signal = row.get('buy_signal', 0)
                value_score = row.get('value_score', 0)
                support_score = row.get('support_score', 0)
                rsi = row.get('rsi', 'N/A')
                
                md_content.append(f"| {ticker} | {name} | {sector} | ${price:.2f} | **{buy_signal:.1f}** | {value_score:.1f} | {support_score:.1f} | {rsi} |")
            
            md_content.append("")
            md_content.append("---")
            md_content.append(f"*AlphaIntelligence Capital | Systematic Alpha Research | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
            
            newsletter_md = "\n".join(md_content)
            newsletter_html = self._markdown_to_html(newsletter_md)
            
            # Create subject
            if not subject:
                subject = f"📊 AlphaIntelligence Capital — Signal Report | {date_str}"
            
            # Create email
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            
            # Add multipart body (plain text + HTML)
            plain_part = MIMEText(newsletter_md, 'plain', 'utf-8')
            html_part = MIMEText(newsletter_html, 'html', 'utf-8')
            msg.attach(plain_part)
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"✅ Screening results sent to {self.recipient_email}")
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"❌ SMTP Auth failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to send screening results: {type(e).__name__}: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test email connection by sending a test email.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self.enabled:
            logger.warning("Email not configured - cannot test")
            return False
        
        try:
            subject = f"🏦 AlphaIntelligence Capital — System Test | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            body = """
            <html>
            <body style="font-family: 'Helvetica Neue', Arial, sans-serif; background: #0a0e27; color: #e0e0e0; padding: 40px;">
                <div style="max-width: 600px; margin: 0 auto; background: #131836; border-radius: 12px; padding: 40px; border: 1px solid #1e2a5a;">
                    <h1 style="color: #c9a84c; font-size: 24px; margin-top: 0;">🏦 AlphaIntelligence Capital</h1>
                    <h2 style="color: #4ade80;">✅ System Connection Verified</h2>
                    <p>Your AlphaIntelligence Capital email delivery pipeline is operational.</p>
                    <table style="width: 100%; margin: 20px 0; border-collapse: collapse;">
                        <tr><td style="padding: 8px 0; color: #9ca3af;">Sender</td><td style="padding: 8px 0; color: #fff;">{sender}</td></tr>
                        <tr><td style="padding: 8px 0; color: #9ca3af;">Recipient</td><td style="padding: 8px 0; color: #fff;">{recipient}</td></tr>
                        <tr><td style="padding: 8px 0; color: #9ca3af;">Timestamp</td><td style="padding: 8px 0; color: #fff;">{time}</td></tr>
                    </table>
                    <hr style="border: none; border-top: 1px solid #1e2a5a; margin: 20px 0;">
                    <p style="color: #6b7280; font-size: 11px; text-align: center;">
                        AlphaIntelligence Capital | Systematic Alpha Research<br>
                        This is an automated system verification.
                    </p>
                </div>
            </body>
            </html>
            """.format(
                sender=self.sender_email,
                recipient=self.recipient_email,
                time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg.attach(MIMEText(body, 'html'))
            
            logger.info(f"Connecting to {self.smtp_server}:{self.smtp_port} for test email...")
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                logger.info(f"Authenticating as {self.sender_email}...")
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"✅ Test email sent successfully to {self.recipient_email}")
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"❌ SMTP Auth FAILED: {e}")
            logger.error(f"   Ensure EMAIL_PASSWORD is a Gmail App Password, not your regular password.")
            return False
        except Exception as e:
            logger.error(f"❌ Test email failed: {type(e).__name__}: {e}")
            return False
    
    def send_error_alert(self, error_message: str, error_details: Optional[str] = None) -> bool:
        """Send error alert email.
        
        Args:
            error_message: Brief error description
            error_details: Full error traceback/details
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False
        
        try:
            subject = f"🚨 AlphaIntelligence Capital — System Alert | {datetime.now().strftime('%Y-%m-%d')}"
            
            body = f"""
            <html>
            <body style="font-family: 'Helvetica Neue', Arial, sans-serif; background: #0a0e27; color: #e0e0e0; padding: 40px;">
                <div style="max-width: 600px; margin: 0 auto; background: #131836; border-radius: 12px; padding: 40px; border: 1px solid #1e2a5a;">
                    <h1 style="color: #c9a84c; font-size: 20px; margin-top: 0;">🏦 AlphaIntelligence Capital</h1>
                    <h2 style="color: #ef4444;">⚠️ System Alert</h2>
                    <p><strong style="color: #9ca3af;">Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong style="color: #9ca3af;">Error:</strong> {error_message}</p>
                    
                    {f'<h3 style="color: #c9a84c;">Details:</h3><pre style="background: #0a0e27; color: #e0e0e0; padding: 15px; border-radius: 8px; overflow-x: auto; border: 1px solid #1e2a5a;">{error_details}</pre>' if error_details else ''}
                    
                    <hr style="border: none; border-top: 1px solid #1e2a5a; margin: 20px 0;">
                    <p style="color: #6b7280; font-size: 11px; text-align: center;">
                        AlphaIntelligence Capital | Automated System Monitor
                    </p>
                </div>
            </body>
            </html>
            """
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"✅ Error alert sent to {self.recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send error alert: {e}")
            return False
    
    def _markdown_to_html(self, markdown_text: str) -> str:
        """Convert markdown to HTML and wrap it in the light newsletter template."""
        import re

        lines = markdown_text.split('\n')
        html_lines = []
        in_table = False
        is_first_table_row = False
        in_list = False

        for line in lines:
            stripped = line.strip()

            if re.match(r'^\|[\s\-:|]+\|$', stripped):
                is_first_table_row = False
                continue

            if stripped.startswith('|') and stripped.endswith('|'):
                if not in_table:
                    if in_list:
                        html_lines.append('</ul>')
                        in_list = False
                    in_table = True
                    is_first_table_row = True
                    html_lines.append('<div class="table-wrap"><table>')
                cells = [cell.strip() for cell in stripped.split('|')[1:-1]]
                tag = 'th' if is_first_table_row else 'td'
                html_lines.append('<tr>' + ''.join(f'<{tag}>{cell}</{tag}>' for cell in cells) + '</tr>')
                continue

            if in_table:
                html_lines.append('</table></div>')
                in_table = False
                is_first_table_row = False

            if stripped.startswith('### '):
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append(f'<h3>{stripped[4:]}</h3>')
                continue
            if stripped.startswith('## '):
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append(f'<h2>{stripped[3:]}</h2>')
                continue
            if stripped.startswith('# '):
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append(f'<h1>{stripped[2:]}</h1>')
                continue
            if stripped == '---':
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                html_lines.append('<hr>')
                continue

            if stripped.startswith('- '):
                if not in_list:
                    in_list = True
                    html_lines.append('<ul>')
                html_lines.append(f'<li>{stripped[2:]}</li>')
                continue
            if in_list and not stripped.startswith('- '):
                html_lines.append('</ul>')
                in_list = False

            if stripped.startswith('> '):
                html_lines.append(f'<blockquote>{stripped[2:]}</blockquote>')
                continue
            if not stripped:
                continue

            html_lines.append(f'<p>{stripped}</p>')

        if in_table:
            html_lines.append('</table></div>')
        if in_list:
            html_lines.append('</ul>')

        html = '\n'.join(html_lines)
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        html = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', html)

        template_path = Path('./src/templates/newsletter_light.html')
        if template_path.exists():
            template = template_path.read_text(encoding='utf-8')
            fallback_card = f'<section class="card">{html}</section>'
            return (
                template.replace('{{hero_headline}}', fallback_card)
                .replace('{{market_mood}}', '')
                .replace('{{indices_strip}}', '')
                .replace('{{sector_table}}', '')
                .replace('{{movers}}', '')
                .replace('{{headlines}}', '')
                .replace('{{events}}', '')
                .replace('{{disclaimer}}', '')
            )

        return f"<html><body>{html}</body></html>"
