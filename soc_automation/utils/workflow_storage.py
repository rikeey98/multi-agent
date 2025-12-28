"""
Workflow storage utility for SOC automation system.

Workflow 실행 결과 저장 유틸리티
- Workflow ID별로 결과 저장
- 각 agent 단계별 결과 저장
- JSON 형식으로 저장
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from soc_automation.utils.logger import get_logger

logger = get_logger()


class WorkflowStorage:
    """
    Workflow 실행 결과를 저장하는 클래스.

    Workflow ID별로 디렉토리를 생성하고 각 단계의 결과를 JSON 파일로 저장합니다.
    """

    def __init__(self, base_dir: str = "workflows"):
        """
        Initialize workflow storage.

        Args:
            base_dir: Base directory for storing workflow results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_workflow_dir(self, workflow_id: str) -> Path:
        """Get workflow directory path."""
        workflow_dir = self.base_dir / workflow_id
        workflow_dir.mkdir(parents=True, exist_ok=True)
        return workflow_dir

    def save_step(
        self,
        workflow_id: str,
        step_name: str,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Save a workflow step result.

        Args:
            workflow_id: Workflow ID
            step_name: Step name (e.g., 'error_analyzer', 'sop_searcher')
            data: Step result data
            timestamp: Timestamp (defaults to now)
        """
        try:
            workflow_dir = self._get_workflow_dir(workflow_id)

            if timestamp is None:
                timestamp = datetime.now()

            # Prepare step data
            step_data = {
                "workflow_id": workflow_id,
                "step": step_name,
                "timestamp": timestamp.isoformat(),
                "data": data
            }

            # Save step file
            step_file = workflow_dir / f"{step_name}.json"
            with open(step_file, 'w', encoding='utf-8') as f:
                json.dump(step_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {step_name} result for workflow {workflow_id}")

        except Exception as e:
            logger.error(f"Failed to save step {step_name}: {e}", exc_info=True)

    def save_complete(
        self,
        workflow_id: str,
        final_state: Dict[str, Any]
    ) -> None:
        """
        Save complete workflow result.

        Args:
            workflow_id: Workflow ID
            final_state: Final workflow state
        """
        try:
            workflow_dir = self._get_workflow_dir(workflow_id)

            # Prepare complete data
            complete_data = {
                "workflow_id": workflow_id,
                "completed_at": datetime.now().isoformat(),
                "final_state": final_state
            }

            # Save complete file
            complete_file = workflow_dir / "complete.json"
            with open(complete_file, 'w', encoding='utf-8') as f:
                json.dump(complete_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved complete workflow result for {workflow_id}")

            # Create summary
            self._create_summary(workflow_id, final_state)

        except Exception as e:
            logger.error(f"Failed to save complete result: {e}", exc_info=True)

    def _create_summary(self, workflow_id: str, final_state: Dict[str, Any]) -> None:
        """Create a human-readable summary file."""
        try:
            workflow_dir = self._get_workflow_dir(workflow_id)

            # Extract key information
            error_analysis = final_state.get('error_analysis', {})
            resolution_plan = final_state.get('resolution_plan', {})
            execution_result = final_state.get('execution_result')

            # Create summary text
            summary_lines = [
                f"=== Workflow Summary: {workflow_id} ===",
                f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "--- Trigger ---",
                f"Event: {final_state.get('trigger_event', 'N/A')}",
                f"Log File: {final_state.get('log_file_path', 'N/A')}",
                "",
                "--- Error Analysis ---",
                f"Error Type: {error_analysis.get('error_type', 'N/A')}",
                f"Severity: {error_analysis.get('severity', 'N/A')}/10",
                f"Message: {error_analysis.get('error_message', 'N/A')}",
                f"Location: {error_analysis.get('location', 'N/A')}",
                "",
                "--- Resolution Plan ---",
                f"Auto-executable: {resolution_plan.get('auto_executable', False)}",
                f"Risk Level: {resolution_plan.get('risk_level', 'N/A')}",
                f"Root Cause: {resolution_plan.get('root_cause', 'N/A')}",
            ]

            # Add resolution steps
            steps = resolution_plan.get('resolution_steps', [])
            if steps:
                summary_lines.append("")
                summary_lines.append("Resolution Steps:")
                for i, step in enumerate(steps, 1):
                    summary_lines.append(f"  {i}. {step}")

            # Add execution result
            if execution_result:
                summary_lines.append("")
                summary_lines.append("--- Execution Result ---")
                summary_lines.append(f"Status: {execution_result.get('status', 'N/A')}")
                if execution_result.get('execution_time'):
                    summary_lines.append(f"Execution Time: {execution_result['execution_time']:.2f}s")

            # Save summary
            summary_file = workflow_dir / "summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_lines))

        except Exception as e:
            logger.error(f"Failed to create summary: {e}", exc_info=True)

    def load_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Load complete workflow result.

        Args:
            workflow_id: Workflow ID

        Returns:
            Workflow data or None if not found
        """
        try:
            workflow_dir = self._get_workflow_dir(workflow_id)
            complete_file = workflow_dir / "complete.json"

            if not complete_file.exists():
                return None

            with open(complete_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Failed to load workflow {workflow_id}: {e}", exc_info=True)
            return None

    def list_workflows(self) -> list:
        """
        List all workflow IDs.

        Returns:
            List of workflow IDs
        """
        try:
            if not self.base_dir.exists():
                return []

            workflows = []
            for item in self.base_dir.iterdir():
                if item.is_dir():
                    workflows.append(item.name)

            return sorted(workflows, reverse=True)  # Latest first

        except Exception as e:
            logger.error(f"Failed to list workflows: {e}", exc_info=True)
            return []


# Global storage instance
_storage: Optional[WorkflowStorage] = None


def get_workflow_storage(base_dir: str = "workflows") -> WorkflowStorage:
    """
    Get or create workflow storage instance.

    Args:
        base_dir: Base directory for storing workflows

    Returns:
        WorkflowStorage instance
    """
    global _storage
    if _storage is None:
        _storage = WorkflowStorage(base_dir)
    return _storage
