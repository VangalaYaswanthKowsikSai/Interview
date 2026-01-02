def calculate_cost(model_config, monthly_tokens=10_000_000):
    """
    Estimates monthly cost based on 10M tokens/month volume.
    """
    # Check if it's an API model or Local model
    if model_config['type'] == 'api':
        # Simple token math
        # Default to 0.02 if not specified (OpenAI small price)
        price = model_config.get('cost_per_1m_tokens', 0.02)
        cost = (monthly_tokens / 1_000_000) * price
        return {
            "cost_usd": cost,
            "note": "Pay-per-use API"
        }
    else:
        # Local estimation (Assuming AWS g4dn.xlarge ~ $0.526/hr)
        server_cost_hourly = 0.526
        server_monthly = server_cost_hourly * 24 * 30
        return {
            "cost_usd": server_monthly,
            "note": "Self-Hosted Server"
        }