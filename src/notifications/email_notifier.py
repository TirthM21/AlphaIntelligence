"""Email notification system for sending newsletters and alerts.

Uses Gmail SMTP (free) to send professional HTML emails.
No credit card required - just a Gmail account with App Password.
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
    """Send professional emails via Gmail SMTP (free)."""
    
    def __init__(self):
        """Initialize email notifier with SMTP settings."""
        self.smtp_server = os.getenv('EMAIL_SMTP_SERVER', "smtp.gmail.com")
        try:
            self.smtp_port = int(os.getenv('EMAIL_SMTP_PORT', 587))
        except (ValueError, TypeError):
            self.smtp_port = 587
        
        # Get credentials from environment
        self.sender_email = os.getenv('EMAIL_SENDER')
        self.sender_password = os.getenv('EMAIL_PASSWORD')  # Gmail App Password or SMTP password
        if self.sender_password:
            self.sender_password = self.sender_password.replace(" ", "")
            
        self.recipient_email = os.getenv('EMAIL_RECIPIENT', self.sender_email)
        
        self.enabled = bool(self.sender_email and self.sender_password)
        
        if not self.enabled:
            missing = []
            if not self.sender_email: missing.append("EMAIL_SENDER")
            if not self.sender_password: missing.append("EMAIL_PASSWORD")
            logger.warning(f"Email notifications disabled - Missing: {', '.join(missing)}")
        else:
            logger.info(f"Email notifications enabled - {self.smtp_server}:{self.smtp_port} - Sending to {self.recipient_email}")
            # Log masked password length for debugging
            pass_len = len(self.sender_password) if self.sender_password else 0
            logger.debug(f"Email config: Sender={self.sender_email}, Password length={pass_len}")
    
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
            
            # Convert markdown to HTML (simple conversion)
            newsletter_html = self._markdown_to_html(newsletter_md)
            
            # Create subject
            if not subject:
                date_str = datetime.now().strftime('%B %d, %Y')
                subject = f"📈 Daily Market Pulse - {date_str}"
            
            # Create email
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"Alpha Intelligence <{self.sender_email}>"
            msg['To'] = self.recipient_email
            
            # Add HTML body
            html_part = MIMEText(newsletter_html, 'html')
            msg.attach(html_part)
            
            # Attach full scan report if provided
            if scan_report_path and Path(scan_report_path).exists():
                logger.debug(f"Attaching scan report: {scan_report_path}")
                try:
                    with open(scan_report_path, 'r', encoding='utf-8') as f:
                        attachment = MIMEText(f.read(), 'plain', 'utf-8')
                        attachment.add_header(
                            'Content-Disposition',
                            'attachment',
                            filename='full_scan_report.txt'
                        )
                        msg.attach(attachment)
                except Exception as attach_err:
                    logger.warning(f"Failed to attach report: {attach_err}")
            
            # Send email
            logger.info(f"Connecting to SMTP server {self.smtp_server}:{self.smtp_port} for {self.recipient_email}...")
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30) as server:
                server.set_debuglevel(1) if os.getenv('LOG_LEVEL') == 'DEBUG' else server.set_debuglevel(0)
                server.starttls()
                logger.info(f"Attempting SMTP login for {self.sender_email}...")
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"✅ SUCCESS: Newsletter sent to {self.recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ SMTP FAILURE for {self.recipient_email}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
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
            subject = f"🚨 Stock Screener Error Alert - {datetime.now().strftime('%Y-%m-%d')}"
            
            body = f"""
            <html>
            <body style="font-family: Arial, sans-serif;">
                <h2 style="color: #d32f2f;">⚠️ Error Alert</h2>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Error:</strong> {error_message}</p>
                
                {f'<h3>Details:</h3><pre style="background: #f5f5f5; padding: 10px; overflow-x: auto;">{error_details}</pre>' if error_details else ''}
                
                <hr>
                <p style="color: #666; font-size: 12px;">
                    This is an automated alert from your Stock Screener system.
                </p>
            </body>
            </html>
            """
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"✅ Error alert sent to {self.recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send error alert: {e}")
            return False

    def send_screening_results(self, results, top_n: int = 20) -> bool:
        """Send screening results via email.
        
        Args:
            results: DataFrame or list of results
            top_n: Number of top results to include
            
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False
            
        try:
            import pandas as pd
            if isinstance(results, pd.DataFrame):
                results_list = results.to_dict('records')
            else:
                results_list = results
                
            top_results = results_list[:top_n]
            
            subject = f"🎯 Stock Screening Results - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Simple HTML table for results
            table_rows = []
            for item in top_results:
                ticker = item.get('ticker', 'N/A')
                score = item.get('score', 0)
                phase = item.get('phase', 'N/A')
                price = item.get('current_price', item.get('price', 0))
                
                table_rows.append(f"""
                <tr>
                    <td><strong>{ticker}</strong></td>
                    <td>{score}</td>
                    <td>{phase}</td>
                    <td>${price:.2f}</td>
                </tr>
                """)
            
            body_html = f"""
            <h2>Stock Screening Results</h2>
            <p>Found {len(results_list)} candidates. Here are the top {len(top_results)}:</p>
            <table border="1" cellpadding="10" style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f2f2f2;">
                    <th>Ticker</th>
                    <th>Score</th>
                    <th>Phase</th>
                    <th>Price</th>
                </tr>
                {"".join(table_rows)}
            </table>
            <p>Check the full report for more details.</p>
            """
            
            # Use newsletter template format
            full_html = self._markdown_to_html(body_html)
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg.attach(MIMEText(full_html, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"✅ Screening results sent to {self.recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send screening results: {e}")
            return False
    
    def _markdown_to_html(self, markdown_text: str) -> str:
        """Convert markdown to HTML (simple conversion for newsletters).
        
        Args:
            markdown_text: Markdown content
            
        Returns:
            HTML string
        """
        html = markdown_text
        
        # Headers
        html = html.replace('# ', '<h1>').replace('\n## ', '</h1>\n<h2>')
        html = html.replace('\n### ', '</h2>\n<h3>')
        
        # Bold
        import re
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        
        # Links
        html = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', html)
        
        # Lists
        html = re.sub(r'\n- (.+)', r'\n<li>\1</li>', html)
        html = html.replace('<li>', '<ul><li>').replace('</li>\n', '</li></ul>\n')
        
        # Tables (basic support)
        lines = html.split('\n')
        in_table = False
        result = []
        
        for line in lines:
            if '|' in line and not line.strip().startswith('|---'):
                if not in_table:
                    result.append('<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">')
                    in_table = True
                
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                result.append('<tr>')
                for cell in cells:
                    result.append(f'<td>{cell}</td>')
                result.append('</tr>')
            elif in_table and '|' not in line:
                result.append('</table>')
                in_table = False
                result.append(line)
            elif not line.strip().startswith('|---'):
                result.append(line)
        
        if in_table:
            result.append('</table>')
        
        html = '\n'.join(result)
        
        # Wrap in HTML template
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {
                    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                    line-height: 1.6;
                    color: #2c3e50;
                    margin: 0;
                    padding: 0;
                    background-color: #f8f9fa;
                }
                .wrapper {
                    padding: 40px 20px;
                }
                .container {
                    background: white;
                    padding: 40px;
                    border-radius: 12px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
                    max-width: 800px;
                    margin: 0 auto;
                }
                h1 {
                    color: #1a2a3a;
                    border-bottom: 4px solid #3498db;
                    padding-bottom: 15px;
                    font-size: 28px;
                    margin-top: 0;
                }
                h2 {
                    color: #2980b9;
                    margin-top: 40px;
                    border-bottom: 1px solid #edf2f7;
                    padding-bottom: 10px;
                    font-size: 22px;
                }
                h3 {
                    color: #34495e;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                    margin-top: 25px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 25px 0;
                    background: #ffffff;
                }
                th, td {
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #edf2f7;
                }
                th {
                    background-color: #f8f9fa;
                    color: #7f8c8d;
                    text-transform: uppercase;
                    font-size: 12px;
                    font-weight: 600;
                }
                ul {
                    background: #fdfdfd;
                    padding: 20px 20px 20px 40px;
                    border-radius: 8px;
                    border-left: 4px solid #ecf0f1;
                    list-style-type: square;
                }
                li {
                    margin-bottom: 10px;
                }
                a {
                    color: #3498db;
                    text-decoration: none;
                    font-weight: 500;
                }
                a:hover {
                    text-decoration: underline;
                }
                blockquote {
                    background: #f1f8ff;
                    border-radius: 8px;
                    padding: 15px 25px;
                    margin: 25px 0;
                    border-left: 5px solid #3498db;
                    font-style: italic;
                    color: #2c3e50;
                }
                .footer {
                    margin-top: 50px;
                    text-align: center;
                    color: #95a5a6;
                    font-size: 12px;
                    border-top: 1px solid #edf2f7;
                    padding-top: 30px;
                }
                strong { color: #1a2a3a; }
            </style>
        </head>
        <body>
            <div class="wrapper">
                <div class="container">
                    {html}
                    <div class="footer">
                        AlphaIntelligence Engine | Advanced Quantitative Screening | 
                        <a href="#">Repository</a> | <a href="#">Dashboard</a>
                        <br><br>
                        © 2026 Your Private Equity Bot. All rights reserved.
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
