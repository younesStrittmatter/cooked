from flask import request, jsonify

def register_ai_only_engine_routes(app, session_manager):
    @app.route("/join", methods=["POST"])
    def join():
        # For AI-only sessions, we don't need to handle human player joins
        # Just return a dummy response since the session is already started
        return jsonify({
            "agent_id": "ai_only_mode",
            "session_id": session_manager.session_id,
            "message": "AI-only mode - session already started"
        })

    @app.route("/intent", methods=["POST"])
    def post_intent():
        data = request.get_json()
        agent_id = data.get("agent_id")
        session_id = data.get("session_id")
        action = data.get("action")

        session = session_manager.get_session(session_id)
        if session:
            session.engine.submit_intent(agent_id, action)
            return jsonify({"status": "ok"})

        return jsonify({"error": "Invalid session"}), 404

    @app.route("/state", methods=["GET"])
    def get_state():
        agent_id = request.args.get("agent_id")
        session_id = request.args.get("session_id")

        session = session_manager.get_session(session_id)
        if session:
            game = session.engine.game
            engine = session.engine
            modules = session.ui_modules

            payload = {
                "you": agent_id,
                "tick": engine.tick_count
            }

            for module in modules:
                payload.update(module.serialize_for_agent(game, engine, agent_id))
            return jsonify(payload)

        return jsonify({"error": "Invalid session"}), 404 