"""Perception: what the duck's head camera sees, turned into facts and events.

    owner_state  the form the VLM fills in for one photo (observable facts only)
    events       a sequence of observed states -> events (came home, stood up, leaving, toy out)
    geometry     a pixel on the floor -> bearing and distance from the robot
    vision_world camera + Grounding DINO + VLM check + geometry, as a `world` for behaviors
    loop         the watching loop: detector every 0.5 s, VLM when the image changes, events, the sentence for the LLM
    async_loop   the same loop with the VLM in the background (capture vs. arrival time, stale answers)
"""
