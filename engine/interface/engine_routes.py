from flask import request, jsonify

def register_engine_routes(app, session_manager):
    @app.route("/join", methods=["POST"])
    def join():
        agent_id = session_manager.generate_agent_id()
        session = session_manager.find_or_create_session()
        session.add_agent(agent_id)

        if session.is_full(session_manager.n_players):
            session.start()

        return jsonify({
            "agent_id": agent_id,
            "session_id": session.id
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

            session._runner.ping()

            payload = {
                "you": agent_id,
                "tick": engine.tick_count
            }

            for module in modules:
                payload.update(module.serialize_for_agent(game, engine, agent_id))
            return jsonify(payload)

        return jsonify({"error": "Invalid session"}), 404
