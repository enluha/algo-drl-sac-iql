from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def load_iql_actor_to_sac(iql_agent, sac_agent):
    """Copy encoder+actor weights from an IQL agent to SAC."""

    src_impl = getattr(iql_agent, "impl", None)
    tgt_impl = getattr(sac_agent, "impl", None)
    if src_impl is None or tgt_impl is None:
        logger.warning("Missing impl attributes on agents; skipping weight bridge")
        return sac_agent

    src_actor = getattr(src_impl, "policy", None) or getattr(src_impl, "actor", None)
    tgt_actor = getattr(tgt_impl, "policy", None) or getattr(tgt_impl, "actor", None)
    if src_actor is None or tgt_actor is None:
        logger.warning("Unable to locate actor modules; skipping weight bridge")
        return sac_agent

    try:
        tgt_actor.load_state_dict(src_actor.state_dict(), strict=False)
        logger.info("Loaded IQL actor weights into SAC policy")
    except RuntimeError as exc:
        logger.warning("Weight bridge failed: %s", exc)
    return sac_agent
