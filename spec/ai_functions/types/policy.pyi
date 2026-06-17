"""Tool-permission policy alias."""

from __future__ import annotations

from typing import Literal

Policy = Literal["allow_always", "deny", "prompt", "allow_session"]
"""Effective policy for a tool call.

``allow_always``
    The tool may run without an approval request.
``deny``
    The tool call is refused without prompting.
``prompt``
    An ``APPROVAL_REQUEST`` is emitted and the executor awaits a decision.
``allow_session``
    Allowed for the current session (see ``PermissionStore.grant_session`` /
    ``revoke_session``).
"""
