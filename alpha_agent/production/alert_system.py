"""
Alert System for monitoring and notifications
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import logging
from typing import Dict, List, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Alert:
    """Represents a single alert"""
    
    def __init__(
        self,
        severity: str,  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
        message: str,
        action: str,
        data: Optional[Dict] = None
    ):
        self.severity = severity
        self.message = message
        self.action = action
        self.data = data or {}
        self.timestamp = datetime.now()
        self.id = f"{self.timestamp.strftime('%Y%m%d%H%M%S')}_{hash(message) % 10000}"
    
    def __repr__(self):
        return f"Alert[{self.severity}]: {self.message} | Action: {self.action}"


class AlertSystem:
    """
    Monitors trading system and sends alerts
    """
    
    def __init__(
        self,
        email_config: Optional[Dict] = None,
        telegram_config: Optional[Dict] = None
    ):
        """
        Args:
            email_config: Email settings {'smtp_server', 'port', 'username', 'password', 'to_email'}
            telegram_config: Telegram bot settings {'bot_token', 'chat_id'}
        """
        self.email_config = email_config
        self.telegram_config = telegram_config
        
        self.alerts = []
        self.alert_rules = self._initialize_rules()
        
        logger.info("AlertSystem initialized")
    
    def _initialize_rules(self) -> Dict:
        """
        Initialize alert rules
        
        Returns:
            Dictionary of alert rules
        """
        return {
            'max_drawdown': {
                'threshold': 0.15,
                'severity': 'HIGH',
                'message_template': 'Drawdown exceeded {threshold}%: current {value}%'
            },
            'daily_loss': {
                'threshold': 0.05,
                'severity': 'HIGH',
                'message_template': 'Daily loss exceeded {threshold}%: {value}%'
            },
            'position_limit': {
                'threshold': 0.40,
                'severity': 'MEDIUM',
                'message_template': 'Position size exceeded {threshold}%: {ticker} at {value}%'
            },
            'regime_change': {
                'threshold': 0.80,
                'severity': 'MEDIUM',
                'message_template': 'Market regime changed to {value} (confidence: {confidence}%)'
            },
            'unusual_activity': {
                'threshold': 3.0,  # Standard deviations
                'severity': 'HIGH',
                'message_template': 'Unusual trading activity detected: {description}'
            },
            'model_degradation': {
                'threshold': 0.5,  # Sharpe ratio drop
                'severity': 'CRITICAL',
                'message_template': 'Model performance degraded: Sharpe {old} → {new}'
            }
        }
    
    def check_drawdown(self, current_drawdown: float):
        """
        Check if drawdown exceeds threshold
        
        Args:
            current_drawdown: Current drawdown value (0-1)
        """
        rule = self.alert_rules['max_drawdown']
        if current_drawdown >= rule['threshold']:
            message = rule['message_template'].format(
                threshold=rule['threshold']*100,
                value=current_drawdown*100
            )
            alert = Alert(
                severity=rule['severity'],
                message=message,
                action="Consider reducing positions or stopping trading",
                data={'drawdown': current_drawdown}
            )
            self.add_alert(alert)
    
    def check_daily_loss(self, daily_return: float):
        """
        Check if daily loss exceeds threshold
        
        Args:
            daily_return: Daily return (negative if loss)
        """
        if daily_return < 0:
            loss = abs(daily_return)
            rule = self.alert_rules['daily_loss']
            if loss >= rule['threshold']:
                message = rule['message_template'].format(
                    threshold=rule['threshold']*100,
                    value=loss*100
                )
                alert = Alert(
                    severity=rule['severity'],
                    message=message,
                    action="Daily loss limit reached - stop trading for the day",
                    data={'daily_return': daily_return}
                )
                self.add_alert(alert)
    
    def check_position_size(self, ticker: str, position_ratio: float):
        """
        Check if position size exceeds limit
        
        Args:
            ticker: Stock ticker
            position_ratio: Position as fraction of portfolio
        """
        rule = self.alert_rules['position_limit']
        if position_ratio >= rule['threshold']:
            message = rule['message_template'].format(
                threshold=rule['threshold']*100,
                ticker=ticker,
                value=position_ratio*100
            )
            alert = Alert(
                severity=rule['severity'],
                message=message,
                action=f"Reduce {ticker} position below {rule['threshold']*100}%",
                data={'ticker': ticker, 'position_ratio': position_ratio}
            )
            self.add_alert(alert)
    
    def check_regime_change(self, new_regime: str, confidence: float):
        """
        Alert on market regime change
        
        Args:
            new_regime: New regime detected
            confidence: Confidence level
        """
        rule = self.alert_rules['regime_change']
        if confidence >= rule['threshold']:
            message = rule['message_template'].format(
                value=new_regime,
                confidence=confidence*100
            )
            alert = Alert(
                severity=rule['severity'],
                message=message,
                action=f"Agent adapting to {new_regime} regime",
                data={'regime': new_regime, 'confidence': confidence}
            )
            self.add_alert(alert)
    
    def check_model_performance(self, old_sharpe: float, new_sharpe: float):
        """
        Check if model performance degraded
        
        Args:
            old_sharpe: Previous Sharpe ratio
            new_sharpe: Current Sharpe ratio
        """
        degradation = old_sharpe - new_sharpe
        rule = self.alert_rules['model_degradation']
        if degradation >= rule['threshold']:
            message = rule['message_template'].format(
                old=old_sharpe,
                new=new_sharpe
            )
            alert = Alert(
                severity=rule['severity'],
                message=message,
                action="Consider retraining model with recent data",
                data={'old_sharpe': old_sharpe, 'new_sharpe': new_sharpe}
            )
            self.add_alert(alert)
    
    def add_alert(self, alert: Alert):
        """
        Add alert and send notifications
        
        Args:
            alert: Alert object
        """
        self.alerts.append(alert)
        logger.warning(f"🚨 {alert}")
        
        # Send notifications
        if self.email_config:
            self._send_email(alert)
        
        if self.telegram_config:
            self._send_telegram(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def _send_email(self, alert: Alert):
        """
        Send email notification
        
        Args:
            alert: Alert to send
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['username']
            msg['To'] = self.email_config['to_email']
            msg['Subject'] = f"[{alert.severity}] AlphaAgent Alert"
            
            body = f"""
            Severity: {alert.severity}
            Time: {alert.timestamp}
            
            Message: {alert.message}
            
            Action Required: {alert.action}
            
            Data: {json.dumps(alert.data, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {alert.message[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    def _send_telegram(self, alert: Alert):
        """
        Send Telegram notification
        
        Args:
            alert: Alert to send
        """
        try:
            import requests
            
            bot_token = self.telegram_config['bot_token']
            chat_id = self.telegram_config['chat_id']
            
            severity_emoji = {
                'LOW': 'ℹ️',
                'MEDIUM': '⚠️',
                'HIGH': '🚨',
                'CRITICAL': '🔴'
            }
            
            message = f"{severity_emoji.get(alert.severity, '⚠️')} *{alert.severity}* Alert\n\n"
            message += f"{alert.message}\n\n"
            message += f"*Action:* {alert.action}"
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=data)
            
            if response.status_code == 200:
                logger.info(f"Telegram alert sent: {alert.message[:50]}...")
            else:
                logger.error(f"Telegram send failed: {response.text}")
                
        except Exception as e:
            logger.error(f"Failed to send Telegram: {e}")
    
    def get_recent_alerts(self, n: int = 10) -> List[Alert]:
        """
        Get recent alerts
        
        Args:
            n: Number of recent alerts
            
        Returns:
            List of alerts
        """
        return self.alerts[-n:]
    
    def export_alerts(self, filepath: str):
        """
        Export alerts to JSON
        
        Args:
            filepath: Path to save
        """
        export_data = []
        for alert in self.alerts:
            export_data.append({
                'id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'severity': alert.severity,
                'message': alert.message,
                'action': alert.action,
                'data': alert.data
            })
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Alerts exported to {filepath}")


if __name__ == "__main__":
    # Demo
    alert_system = AlertSystem()
    
    # Test alerts
    alert_system.check_drawdown(0.18)
    alert_system.check_daily_loss(-0.06)
    alert_system.check_regime_change('BEAR', 0.85)
    
    print("\nRecent Alerts:")
    for alert in alert_system.get_recent_alerts():
        print(f"  {alert}")

