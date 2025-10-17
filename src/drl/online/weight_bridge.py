from __future__ import annotations

from d3rlpy.algos.base import AlgoBase


def load_iql_actor_to_sac(iql_agent: AlgoBase, sac_agent: AlgoBase) -> AlgoBase:
    """Copy policy parameters from an IQL agent into a SAC agent."""
    if iql_agent.impl is None or sac_agent.impl is None:
        raise RuntimeError("Both agents must be built before transferring weights.")
    sac_agent.impl.policy.load_state_dict(iql_agent.impl.policy.state_dict())
    return sac_agent
