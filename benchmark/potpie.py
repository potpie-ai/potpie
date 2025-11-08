# parse a local repo and save it to the local database
# verify and proceed towards multiple repos at the same time with semaphore

from dotenv import load_dotenv
from pathlib import Path
import asyncio
from uuid6 import UUID, uuid7
import sys
import logging
from datetime import datetime, timezone


# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent.parent))


from app.core.database import create_celery_async_session
from app.modules.conversations.conversation.conversation_model import ConversationStatus
from app.modules.parsing.graph_construction.parsing_service import ParsingService
from app.modules.parsing.graph_construction.parsing_schema import ParsingRequest
from app.modules.conversations.conversation.conversation_service import (
    ConversationService,
)
from app.modules.conversations.message.message_model import MessageType
from app.modules.conversations.conversation.conversation_schema import (
    CreateConversationRequest,
)
from app.core.base_model import Base
from app.core.database import SessionLocal, engine

from app.modules.conversations.message.message_schema import MessageRequest
from app.modules.conversations.message.message_store import MessageStore
from app.modules.conversations.conversation.conversation_store import ConversationStore
from app.core.models import *  # noqa #necessary for models to not give import errors
from app.modules.users.user_service import UserService
from app.modules.users.user_schema import CreateUser
from app.core.database import get_db

from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval import evaluate


_ = load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_eval_user() -> str:
    eval_user_id = "eval_user"

    db = SessionLocal()

    try:
        user_service = UserService(db)
        user = user_service.get_user_by_uid(eval_user_id)

        if user:
            logger.info("Evaluation user already exists: %s", eval_user_id)
            return eval_user_id
        else:
            # Create new evaluation user
            user = CreateUser(
                uid=eval_user_id,
                email="eval@potpie.ai",
                display_name="Evaluation User",
                email_verified=True,
                created_at=datetime.now(timezone.utc),
                last_login_at=datetime.now(timezone.utc),
                provider_info={"access_token": "eval_token"},
                provider_username="eval",
            )

            uid, _, error = user_service.create_user(user)

            if error:
                raise Exception("Failed to create evaluation user: %s", error)

            logger.info("Created evaluation user with uid: %s", uid)
            return str(uid)

    finally:
        db.close()


async def parse_single_repo(
    repo_path: Path,
    user_id: str = "benchmark_user",
    user_email: str = "benchmark@potpie.ai",
    project_id: UUID | None = None,
    branch_name: str | None = None,
    commit_id: str | None = None,
) -> UUID:
    if project_id is None:
        project_id = uuid7()

    db_generator = get_db()
    db = next(db_generator)

    try:
        parsing_service = ParsingService(db, user_id=user_id)
        repo_details = ParsingRequest(
            repo_path=str(repo_path), branch_name=branch_name, commit_id=commit_id
        )
        _ = await parsing_service.parse_directory(
            repo_details=repo_details,
            user_id=user_id,
            user_email=user_email,
            project_id=str(project_id),
            cleanup_graph=True,
        )
        return project_id

    except Exception:
        logging.error("Error parsing repository", exc_info=True)
        raise
    finally:
        db.close()


async def create_conversation_and_post_message(
    user_id: str, email_id: str, project_id: str, question: str
) -> str:
    """Create a conversation linked to a parsed repository."""
    db = next(get_db())
    async_session, engine = create_celery_async_session()

    try:
        conversation_store = ConversationStore(db, async_session)
        message_store = MessageStore(db, async_session)

        conversation_service = ConversationService.create(
            conversation_store=conversation_store,
            message_store=message_store,
            db=db,
            user_id=user_id,
            user_email=email_id,
        )

        request = CreateConversationRequest(
            title=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_id=user_id,
            project_ids=[project_id],
            agent_ids=["codebase_qna_agent"],
            status=ConversationStatus.ACTIVE,
        )

        conversation_id, _ = await conversation_service.create_conversation(
            request, user_id
        )

        message_request = MessageRequest(
            content=question,
        )

        full_response = ""
        async for chunk in conversation_service.store_message(
            conversation_id=conversation_id,
            message=message_request,
            message_type=MessageType.HUMAN,
            user_id=user_id,
            stream=False,
        ):
            full_response += chunk.message
        return full_response

    finally:
        await async_session.close()
        await engine.dispose()
        db.close()


async def parse_multiple_repos(repo_paths: list[Path]) -> list[UUID]:
    semaphore = asyncio.BoundedSemaphore(10)

    async def _guarded_parse(repo_path: Path) -> UUID:
        async with semaphore:
            return await parse_single_repo(repo_path)

    tasks = [asyncio.create_task(_guarded_parse(path)) for path in repo_paths]
    results = await asyncio.gather(*tasks)
    return results


async def main():
    Base.metadata.create_all(bind=engine)

    logger.info("Setting up evaluation user...")
    user_id = setup_eval_user()

    repo_path = Path("/home/dsantra/e3nn")
    commit_id = "d7661462f07773b78e9dd1141520d41ab33aed1c"

    # Parse repository
    # logger.info(f"Starting parsing of repository: {repo_path}")
    # project_id = await parse_single_repo(
    #     repo_path=repo_path,
    #     user_id=user_id,
    #     user_email="eval@potpie.ai",
    #     commit_id=commit_id,
    # )
    # project_id = str(project_id)
    # print(f"Parsed repository. Project ID: {project_id}")
    project_id = "019a5d95-8d7c-7c3e-861e-17360414b493"

    # Create conversation to ask questions about the repo
    logger.info("Creating conversation for repository Q&A...")
    response = await create_conversation_and_post_message(
        user_id=user_id,
        email_id="eval@potpie.ai",
        project_id=str(project_id),
        question="How does e3nn achieve E(3)-equivariance through Irreducible Representations (Irreps) and Tensor Product operations?",
    )
    print(response)
    results = evaluate(
        metrics=[AnswerRelevancyMetric()],
        test_cases=[
            LLMTestCase(
                input=response,
                actual_output="""
                e3nn achieves E(3)-equivariance through three key mechanisms:

                ### 1. Mathematical Foundation of Irreducible Representations

                In e3nn, the [`Irrep`](e3nn/o3/_irreps.py#L14) class encapsulates irreducible representations of the O(3) group, where each irrep is defined by parameters `(l, p)`:

                - `l`: The order of the representation (0, 1, 2, ...), determining the dimension `dim = 2l + 1`
                - `p`: Parity (±1), representing behavior under spatial inversion

                For example, `1e` represents an l=1 even-parity vector (ordinary vector), while `1o` represents an l=1 odd-parity pseudovector (like angular momentum).

                ### 2. Equivariant Construction of Tensor Products

                The [`TensorProduct`](e3nn/o3/_tensor_product/_tensor_product.py#L25) class implements equivariant tensor product operations. The core lies in:

                ```python
                # Clebsch-Gordan coefficients ensure equivariance
                def D_from_angles(self, alpha, beta, gamma, k=None):
                    # Generate rotation matrix D^l(alpha, beta, gamma)
                ```

                The tensor product follows Clebsch-Gordan rules: the tensor product of two irreps `l₁` and `l₂` decomposes as:

                ```
                l₁ ⊗ l₂ = ⊕_{l=|l₁-l₂|}^{l₁+l₂} l
                ```

                ### 3. Flexible Instruction System

                Tensor products are controlled through instruction lists:

                ```python
                instructions = [
                    (i_in1, i_in2, i_out, connection_mode, has_weight, path_weight)
                ]
                ```

                Where `connection_mode` defines multiplicity handling:

                - `"uvw"`: Fully connected, each input connects to each output
                - `"uuu"`: Element-wise operation
                - `"uvu"`, `"uvv"`, etc.: Partial connection modes

                This design enables e3nn to construct neural network layers that preserve rotational and translational equivariance, suitable for 3D data processing tasks like molecular modeling and materials science.
                """,
            ),
        ],
    )
    relevant_result = results[0]

    # # Ask some questions about the repository
    # questions = [
    #     "What is this repository about?",
    #     "What are the main components and how do they work together?",
    #     "Can you find any potential bugs or improvements?",
    # ]

    # for i, question in enumerate(questions, 1):
    #     logger.info(f"Question {i}: {question}")
    #     response = ask_about_repo(conversation_id, user_id, question)
    #     print(f"\nQ{i}: {question}")
    #     print(f"A{i}: {response}\n")

    # print("All done!")


if __name__ == "__main__":
    asyncio.run(main())
