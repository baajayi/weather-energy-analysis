"""
Notification system for pipeline failures and status updates.
"""

import logging
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class NotificationSystem:
    """Handle notifications for pipeline failures and status updates."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the notification system."""
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        
        # In a production environment, these would come from environment variables
        # For now, we'll use basic GitHub Issues integration
        self.github_token = None  # Would be set from environment
        self.github_repo = None   # Would be set from config
        
    def create_github_issue(self, title: str, body: str, labels: List[str] = None) -> bool:
        """
        Create a GitHub issue for notification (when token is available).
        
        Args:
            title: Issue title
            body: Issue body content
            labels: List of labels to apply
            
        Returns:
            True if issue created successfully, False otherwise
        """
        if not self.github_token or not self.github_repo:
            self.logger.info("GitHub integration not configured, logging notification instead")
            self._log_notification("GitHub Issue", title, body, labels)
            return False
        
        try:
            url = f"https://api.github.com/repos/{self.github_repo}/issues"
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            data = {
                'title': title,
                'body': body,
                'labels': labels or []
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            issue_url = response.json().get('html_url', 'Unknown')
            self.logger.info(f"Created GitHub issue: {issue_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create GitHub issue: {e}")
            self._log_notification("Failed GitHub Issue", title, body, labels)
            return False
    
    def notify_pipeline_failure(self, error_details: Dict, pipeline_type: str = "daily") -> bool:
        """
        Notify about pipeline failure.
        
        Args:
            error_details: Dictionary containing error information
            pipeline_type: Type of pipeline that failed
            
        Returns:
            True if notification sent successfully
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        title = f"🚨 {pipeline_type.title()} Pipeline Failure - {datetime.now().strftime('%Y-%m-%d')}"
        
        body = f"""
## Pipeline Failure Report

**Timestamp:** {timestamp}
**Pipeline Type:** {pipeline_type}
**Status:** {error_details.get('status', 'error')}

### Error Details
```
{error_details.get('message', 'No error message provided')}
```

### Additional Information
- **Records Processed:** {error_details.get('processed_records', 0)}
- **Data Sources:** {error_details.get('data_sources', 'Unknown')}
- **Quality Score:** {error_details.get('quality_score', 'N/A')}

### Troubleshooting Steps
1. Check API connectivity and credentials
2. Verify data source availability
3. Review pipeline logs for detailed error information
4. Consider running manual backfill if needed

### Action Required
- [ ] Investigate root cause
- [ ] Fix underlying issue
- [ ] Run manual pipeline if necessary
- [ ] Update monitoring/alerting if needed

---
*This issue was automatically created by the data pipeline notification system.*
        """
        
        labels = ['pipeline-failure', 'urgent', pipeline_type]
        
        return self.create_github_issue(title, body, labels)
    
    def notify_data_quality_issue(self, quality_report: Dict, threshold: float = 70.0) -> bool:
        """
        Notify about data quality issues.
        
        Args:
            quality_report: Data quality validation results
            threshold: Quality score threshold for notifications
            
        Returns:
            True if notification sent successfully
        """
        quality_score = quality_report.get('integrity_score', 0)
        
        if quality_score >= threshold:
            return True  # No notification needed for good quality
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        title = f"⚠️ Data Quality Alert - Score: {quality_score:.1f}/100"
        
        warnings = quality_report.get('warnings', [])
        errors = quality_report.get('errors', [])
        
        body = f"""
## Data Quality Alert

**Timestamp:** {timestamp}
**Quality Score:** {quality_score:.1f}/100 (Threshold: {threshold})
**Total Records:** {quality_report.get('total_records', 0)}

### Issues Detected

#### Errors ({len(errors)})
"""
        
        for error in errors:
            body += f"- ❌ {error}\n"
        
        body += f"""
#### Warnings ({len(warnings)})
"""
        
        for warning in warnings:
            body += f"- ⚠️ {warning}\n"
        
        body += f"""
### Quality Checks Summary
"""
        
        checks = quality_report.get('checks', {})
        for check_name, result in checks.items():
            if isinstance(result, str):
                emoji = "✅" if result == "PASS" else "⚠️" if result == "WARNING" else "❌"
                body += f"- {emoji} **{check_name}**: {result}\n"
        
        body += f"""
### Recommended Actions
1. Review data sources for accuracy
2. Check for API or processing issues
3. Validate data transformation logic
4. Consider data cleaning procedures

---
*This alert was automatically generated by the data quality monitoring system.*
        """
        
        labels = ['data-quality', 'warning', 'monitoring']
        
        return self.create_github_issue(title, body, labels)
    
    def notify_missing_data(self, missing_dates: List[str], backfill_results: Dict = None) -> bool:
        """
        Notify about missing data and backfill attempts.
        
        Args:
            missing_dates: List of dates with missing data
            backfill_results: Results from backfill attempt
            
        Returns:
            True if notification sent successfully
        """
        if not missing_dates:
            return True  # No notification needed
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        title = f"📊 Missing Data Detected - {len(missing_dates)} dates"
        
        body = f"""
## Missing Data Report

**Timestamp:** {timestamp}
**Missing Dates:** {len(missing_dates)}

### Dates with Missing Data
"""
        
        for date in missing_dates[:10]:  # Limit to first 10 dates
            body += f"- {date}\n"
        
        if len(missing_dates) > 10:
            body += f"- ... and {len(missing_dates) - 10} more dates\n"
        
        if backfill_results:
            body += f"""
### Backfill Results
- **Status:** {backfill_results.get('status', 'unknown')}
- **Dates Processed:** {len(backfill_results.get('dates_processed', []))}
- **Dates Failed:** {len(backfill_results.get('dates_failed', []))}
- **Total Records:** {backfill_results.get('total_records', 0)}

#### Message
{backfill_results.get('message', 'No message provided')}
"""
        
        body += f"""
### Recommended Actions
1. Check API data availability for missing dates
2. Verify pipeline scheduling and execution
3. Review error logs for failed pipeline runs
4. Consider manual data recovery if needed

---
*This report was automatically generated by the data monitoring system.*
        """
        
        labels = ['missing-data', 'monitoring', 'backfill']
        
        return self.create_github_issue(title, body, labels)
    
    def notify_successful_recovery(self, recovery_details: Dict) -> bool:
        """
        Notify about successful data recovery or issue resolution.
        
        Args:
            recovery_details: Details about the recovery
            
        Returns:
            True if notification sent successfully
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        title = f"✅ Pipeline Recovery Successful - {datetime.now().strftime('%Y-%m-%d')}"
        
        body = f"""
## Pipeline Recovery Report

**Timestamp:** {timestamp}
**Recovery Type:** {recovery_details.get('type', 'automatic')}

### Recovery Details
- **Records Processed:** {recovery_details.get('records_processed', 0)}
- **Data Sources:** {recovery_details.get('data_sources', 'multiple')}
- **Quality Score:** {recovery_details.get('quality_score', 'N/A')}
- **Source:** {recovery_details.get('source', 'primary APIs')}

### Message
{recovery_details.get('message', 'Pipeline has recovered and is operating normally.')}

---
*This notification was automatically generated by the pipeline monitoring system.*
        """
        
        labels = ['pipeline-recovery', 'success', 'monitoring']
        
        return self.create_github_issue(title, body, labels)
    
    def _log_notification(self, notification_type: str, title: str, body: str, labels: List[str] = None):
        """
        Log notification details when external systems are not available.
        
        Args:
            notification_type: Type of notification
            title: Notification title
            body: Notification body
            labels: Notification labels
        """
        self.logger.warning(f"NOTIFICATION [{notification_type}]: {title}")
        
        # Also save to a notifications log file for later review
        try:
            logs_path = Path("logs")
            logs_path.mkdir(exist_ok=True)
            
            notification_log = logs_path / "notifications.jsonl"
            
            notification_record = {
                'timestamp': datetime.now().isoformat(),
                'type': notification_type,
                'title': title,
                'body': body,
                'labels': labels or [],
                'status': 'logged_only'
            }
            
            with open(notification_log, 'a') as f:
                f.write(json.dumps(notification_record) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to log notification: {e}")
    
    def check_notification_backlog(self) -> List[Dict]:
        """
        Check for any unprocessed notifications in the log.
        
        Returns:
            List of notification records
        """
        try:
            notification_log = Path("logs/notifications.jsonl")
            
            if not notification_log.exists():
                return []
            
            notifications = []
            with open(notification_log, 'r') as f:
                for line in f:
                    try:
                        notification = json.loads(line.strip())
                        notifications.append(notification)
                    except json.JSONDecodeError:
                        continue
            
            return notifications
            
        except Exception as e:
            self.logger.error(f"Failed to check notification backlog: {e}")
            return []