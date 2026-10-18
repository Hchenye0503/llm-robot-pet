"""Planner: sentence -> LLM -> validated plan -> policy commands.

    LLM (structured output)
      -> schema.Plan            structure check (unknown actions / fields are rejected)
      -> reverse.expand_backward  "walk backward" -> turn around, walk, turn back
      -> preconditions          insert "stand up" before steps that need standing
      -> translate              each step -> (policy, 13 command numbers); limits.apply_limit clamps or rejects
    llm.plan_with_retry ties it together: rejected -> one rewrite with the error attached -> else degrade.
"""
