# AI-Powered LinkedIn Content Automation Agent

An extensible Python application for researching AI developments, generating and validating LinkedIn content, requesting human approval, and publishing through the official LinkedIn API.

## Phase 1 Status

Phases 1 through 9 provide the project structure, environment-backed configuration, the SQLAlchemy persistence layer, RSS research, LLM integration, topic selection, draft generation, deterministic fact/quality evaluation, LangGraph orchestration, and a Streamlit dashboard with optional daily scheduling. Publishing will be added in a later phase.

## Project Structure

```text
ai-linkedin-agent/
├── agents/             # Research and content-quality agents
├── api/                # FastAPI routes and OAuth handlers
├── database/           # SQLAlchemy models, sessions, and CRUD operations
├── frontend/           # Streamlit dashboard
├── graph/              # LangGraph state and workflow
├── prompts/            # Editable prompt templates
├── scheduler/          # Daily workflow scheduler
├── services/           # LLM, research, embeddings, and LinkedIn services
├── tests/              # Automated tests
├── config.py           # Environment-backed settings
├── main.py             # FastAPI application entry point
└── requirements.txt    # Python dependencies
```

## Setup

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add credentials only when the relevant integrations are implemented. Secrets must never be committed.

## Run the Initial API

```powershell
uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000/health`. The initial service returns `{"status":"ok"}`.

## Planned Workflow

Research -> topic selection -> writing -> fact checking -> quality checking -> duplicate detection -> human approval -> official LinkedIn publishing.

Publishing will remain disabled by default and will never use browser automation or unofficial endpoints.

## Phase 7 Evaluation

Research -> topic selection -> draft generation -> fact checking -> quality checking -> final evaluation. Phase 7 uses supplied source evidence, deterministic quality and security gates, and returns `APPROVED`, `NEEDS_REVISION`, or `REJECTED`. It does not publish posts.

## Phase 8 Workflow

Research Agent -> Topic Selection Agent -> LinkedIn Writer Agent -> Fact Checker Agent -> Quality Checker Agent -> Duplicate Detection Agent -> Decision Router. LangGraph controls state, routing, and the bounded revision loop; the existing services perform the business logic. The workflow allows two revisions by default and never publishes to LinkedIn.

## Phase 9 Dashboard and Scheduler

Install dependencies, then run the dashboard with `streamlit run frontend/dashboard.py`. The dashboard reads existing database records and delegates manual runs to `WorkflowService`, which invokes the compiled LangGraph workflow. Scheduling is disabled by default; enable it with `SCHEDULER_ENABLED=true`, configure `SCHEDULE_TIME` and `SCHEDULE_TIMEZONE`, and restart the dashboard. `MAX_CONCURRENT_WORKFLOWS=1` prevents overlap. LinkedIn publishing is not implemented, and approved posts remain drafts.

## Database

The default development database is SQLite at `./linkedin_agent.db`. Tables are created by calling `database.init_db()`; production deployments can use a PostgreSQL-compatible `DATABASE_URL` without changing the models or CRUD API.

## Phase 10 LinkedIn Integration

LinkedIn integration uses the official OAuth 2.0 authorization-code flow and official APIs:

- Authorization: `https://www.linkedin.com/oauth/v2/authorization`
- Token exchange: `https://www.linkedin.com/oauth/v2/accessToken`
- Member identity: `GET https://api.linkedin.com/v2/userinfo` with `openid`
- Member post: `POST https://api.linkedin.com/v2/ugcPosts` with `w_member_social`
- Required post header: `X-Restli-Protocol-Version: 2.0.0`
- API version header: configured by `LINKEDIN_API_VERSION`

The Share on LinkedIn product and the `w_member_social` permission must be enabled for the LinkedIn developer application. Available scopes depend on the products approved for that application. Configure the exact redirect URI in the LinkedIn Developer Portal and in `.env`.

Set `LINKEDIN_PUBLISHING_ENABLED=false` for normal development. Publishing requires an AI-approved draft and a separate explicit human approval that changes its status to `APPROVED_FOR_PUBLISH`. Tokens are encrypted with Fernet using `LINKEDIN_TOKEN_ENCRYPTION_KEY`; the key must be kept outside Git. Disconnect removes the local connection and does not claim provider-side revocation.

Automated tests use mocked HTTP responses and never call LinkedIn. Real credentials and API access are required for manual end-to-end verification.

## Production Readiness

- Configuration is environment-based and exposes only a safe summary. URLs, scheduler time/timezone, and publishing prerequisites are validated at startup or explicit scheduler activation.
- Structured application events include run, agent, and event context. Credential-like values are redacted from configured logs.
- `GET /health/live` provides a process liveness check, while `GET /health/ready` reports configuration and dependency readiness without secrets.
- Workflow counts and stage durations are tracked by the lightweight internal metrics service. Workflow runs retain safe in-process history and error summaries.
- Scheduler and publishing are disabled by default. Scheduler runs use a configured timezone and lock against overlapping workflows. Publishing still requires human approval and preserves failed drafts when the provider call fails.
- Docker uses Python 3.13, a non-root user, an external `.env`, a persistent data volume, and a container health check. Compose explicitly keeps scheduling and publishing disabled.
- Use `pytest -v` for deterministic tests. Tests use isolated databases, mocked HTTP, and fake providers; they do not require real Gemini or LinkedIn credentials.