def load_iql_actor_to_sac(iql_agent, sac_agent, logger=None):
    # works for d3rlpy agents with compatible encoders
    try:
        sac_actor = sac_agent.impl.policy
        iql_actor = iql_agent.impl.policy
        sac_actor.load_state_dict(iql_actor.state_dict(), strict=False)
        if logger: logger.debug("IQL→SAC actor weights loaded")
    except Exception as e:
        if logger: logger.warning(f"Weight bridge failed: {e}")
