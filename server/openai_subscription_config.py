import os

MODE = 'OH_OPENAI_SUBSCRIPTION_MODE'
BROKER_URL = 'OH_OPENAI_SUBSCRIPTION_BROKER_URL'


def cloud_subscription_enabled() -> bool:
    return os.getenv('OH_ENABLE_CLOUD_OPENAI_SUBSCRIPTION', '').lower() in ('1', 'true')


def configure_runtime_subscription(
    env: dict[str, str], web_url: str | None, *, broker: bool = False
) -> None:
    if not cloud_subscription_enabled():
        return
    env.pop(BROKER_URL, None)
    env[MODE] = 'disabled'
    if broker and web_url:
        env[MODE] = 'broker'
        env[BROKER_URL] = (
            f'{web_url.rstrip("/")}/api/llm/subscription/openai/runtime/token'
        )
