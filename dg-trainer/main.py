from enum import Enum

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent

load_dotenv()

SYSTEM_PROMPT = """You are a disc golf putting coach.

Return a PRACTICE SESSION as JSON matching the provided schema.
Rules:
- Keep total time within the user's requested minutes.
- Use 2 to 5 drills, each with explicit setup, reps, scoring, and a focus cue.
- Do NOT include warm motivational text; be operational.
- If the user provides a list of distances, ONLY use those distances.
- If no distances list is provided, choose sensible distances <= 35ft by default unless the user asks otherwise.
- Make the session realistic for the time and putter count.
"""


class Environment(str, Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"


class PracticeConstraints(BaseModel):
    minutes: int
    putters: int
    distances: list[int]
    environment: str = Field(default=Environment.INDOOR)


class PracticeRequest(BaseModel):
    constraints: PracticeConstraints = Field(default_factory=PracticeConstraints)


class PracticeDrill(BaseModel):
    name: str = Field(description="The name of the drill.")
    minutes: int = Field(description="The minutes of the drill.")
    distance: int = Field(description="The distance of the drill.")
    reps: int = Field(description="The number of reps of the drill.")
    instructions: str = Field(description="The instructions of the drill.")


class PracticeSession(BaseModel):
    total_minutes: int = Field(description="The total minutes of the practice session.")
    focus: str = Field(description="The focus of the practice session.")
    drills: list[PracticeDrill]


agent = Agent(
    "anthropic:claude-sonnet-4-0",
    system_prompt=SYSTEM_PROMPT,
    output_type=PracticeSession,
)


def build_user_prompt(req: PracticeRequest) -> str:
    return (
        "Build a putting practice session for this request:\n"
        f"{req.model_dump_json(indent=2)}"
    )


def main():
    req = PracticeRequest(
        constraints=PracticeConstraints(minutes=10, putters=10, distances=[15, 20, 30]),
    )

    result = agent.run_sync(build_user_prompt(req), deps=req)

    print(result.output)


if __name__ == "__main__":
    main()
