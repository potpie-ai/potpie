import os
import pytest
import subprocess
import time

from potpie.server_manager import (
    ServerManager,
    DockerError,
    PostgresError,
    MigrationError,
    StartServerError,
    StopServerError,
)


@pytest.fixture(scope="module")
def server_manager():
    return ServerManager()


@pytest.mark.parametrize(
    "env_var, expected",
    [
        ("custom", False),
        ("default", False),
        ("production", False),
        ("development", True),
    ],
)
def test_check_environment(server_manager, monkeypatch, env_var, expected):
    monkeypatch.setenv("ENV", env_var)
    assert server_manager.check_environment() == expected


@pytest.mark.parametrize(
    "is_docker_installed, docker_compose_up, docker_ps_output, expected_exception",
    [
        # Case 1: Docker is not installed → should raise DockerError
        (False, None, None, DockerError),
        # Case 2: Docker Compose fails → should raise DockerError
        (True, DockerError, None, DockerError),
        # Case 3: Docker starts but containers are not all healthy → should raise DockerError
        (
            True,
            0,
            """{"State": "\\"excited\\""}\n{"State": "\\"excited\\""}\n{"State": "excited"}\n""",
            DockerError,
        ),
        # Case 4: All containers are running → should pass (No Exception)
        (
            True,
            0,
            """{"Command":"\\"tini -g -- /startup…\\"","CreatedAt":"2025-02-13 23:12:24 +0530 IST","ExitCode":0,"Health":"","ID":"c8e7a2bad4d1","Image":"neo4j:latest","Labels":"com.docker.compose.version=2.32.4,com.docker.compose.container-number=1,com.docker.compose.image=sha256:3b9cf65b6cae8b166d5091969e80a5600028aa2e8676a45a925dcc15c379024e,com.docker.compose.oneoff=False,com.docker.compose.project.config_files=/home/deepesh/Development/public/opensource/pot/potpie/docker-compose.yaml,com.docker.compose.service=neo4j,com.docker.compose.config-hash=d3f4c209daf619afcaae2b5762a1374d15939579c1bc8480e0df362f74e69855,com.docker.compose.depends_on=,com.docker.compose.project=potpie,com.docker.compose.project.working_dir=/home/deepesh/Development/public/opensource/pot/potpie","LocalVolumes":"2","Mounts":"d332d4405c91cb…,92d5ff89378b27…","Name":"potpie_neo4j","Names":"potpie_neo4j","Networks":"potpie_app-network","Ports":"0.0.0.0:7474-\\u003e7474/tcp, :::7474-\\u003e7474/tcp, 7473/tcp, 0.0.0.0:7687-\\u003e7687/tcp, :::7687-\\u003e7687/tcp","Project":"potpie","Publishers":[{"URL":"","TargetPort":7473,"PublishedPort":0,"Protocol":"tcp"},{"URL":"0.0.0.0","TargetPort":7474,"PublishedPort":7474,"Protocol":"tcp"},{"URL":"::","TargetPort":7474,"PublishedPort":7474,"Protocol":"tcp"},{"URL":"0.0.0.0","TargetPort":7687,"PublishedPort":7687,"Protocol":"tcp"},{"URL":"::","TargetPort":7687,"PublishedPort":7687,"Protocol":"tcp"}],"RunningFor":"2 seconds ago","Service":"neo4j","Size":"0B","State":"running","Status":"Up 1 second"}\n{"Command":"\\"docker-entrypoint.s…\\"","CreatedAt":"2025-02-13 23:12:24 +0530 IST","ExitCode":0,"Health":"starting","ID":"f812dc33590a","Image":"postgres:latest","Labels":"com.docker.compose.image=sha256:b781f3a53e61df916d97dffe6669ef32b08515327ee3a398087115385b5178f5,com.docker.compose.version=2.32.4,com.docker.compose.oneoff=False,com.docker.compose.project=potpie,com.docker.compose.project.config_files=/home/deepesh/Development/public/opensource/pot/potpie/docker-compose.yaml,com.docker.compose.project.working_dir=/home/deepesh/Development/public/opensource/pot/potpie,com.docker.compose.service=postgres,com.docker.compose.config-hash=4d1a07dcf14c3ce4dad08e2f5b8c984841ff240fdce1553f168e072f6e4def1d,com.docker.compose.container-number=1,com.docker.compose.depends_on=","LocalVolumes":"1","Mounts":"e5fc461ef93fdd…","Name":"potpie_postgres","Names":"potpie_postgres","Networks":"potpie_app-network","Ports":"0.0.0.0:5432-\\u003e5432/tcp, :::5432-\\u003e5432/tcp","Project":"potpie","Publishers":[{"URL":"0.0.0.0","TargetPort":5432,"PublishedPort":5432,"Protocol":"tcp"},{"URL":"::","TargetPort":5432,"PublishedPort":5432,"Protocol":"tcp"}],"RunningFor":"2 seconds ago","Service":"postgres","Size":"0B","State":"running","Status":"Up 1 second (health: starting)"}\n{"Command":"\\"docker-entrypoint.s…\\"","CreatedAt":"2025-02-13 23:12:24 +0530 IST","ExitCode":0,"Health":"","ID":"3ed7ceaa1fa9","Image":"redis:latest","Labels":"com.docker.compose.container-number=1,com.docker.compose.image=sha256:4075a3f8c3f8e3878d1041c5019e4af445e3b79cf3b55e03063f9813cd49e3f2,com.docker.compose.project=potpie,com.docker.compose.project.config_files=/home/deepesh/Development/public/opensource/pot/potpie/docker-compose.yaml,com.docker.compose.project.working_dir=/home/deepesh/Development/public/opensource/pot/potpie,com.docker.compose.service=redis,com.docker.compose.version=2.32.4,com.docker.compose.config-hash=2d9a2cf61dce1849e37f538d64402d3600894c11250e2a3f40a6b6bc713c3977,com.docker.compose.depends_on=,com.docker.compose.oneoff=False","LocalVolumes":"1","Mounts":"3691d0cf3ee0d3…","Name":"potpie_redis_broker","Names":"potpie_redis_broker","Networks":"potpie_app-network","Ports":"0.0.0.0:6379-\\u003e6379/tcp, :::6379-\\u003e6379/tcp","Project":"potpie","Publishers":[{"URL":"0.0.0.0","TargetPort":6379,"PublishedPort":6379,"Protocol":"tcp"},{"URL":"::","TargetPort":6379,"PublishedPort":6379,"Protocol":"tcp"}],"RunningFor":"2 seconds ago","Service":"redis","Size":"0B","State":"running","Status":"Up 1 second"}\n""",
            None,
        ),
    ],
)
def test_start_docker(
    server_manager,
    monkeypatch,
    is_docker_installed,
    docker_compose_up,
    docker_ps_output,
    expected_exception,
):

    monkeypatch.setattr(
        ServerManager, "is_docker_installed", lambda self: is_docker_installed
    )

    def mock_popen(*args, **kwargs):
        class MockProcess:
            def __init__(self, returncode):
                self.returncode = returncode
                self.stdout = ""
                self.stderr = ""

            def poll(self):
                return True

        if docker_compose_up is DockerError:
            raise DockerError("Docker Compose failed")
        return MockProcess(docker_compose_up)

    monkeypatch.setattr(subprocess, "Popen", mock_popen)

    def mock_run(command, capture_output=True, text=True):
        return subprocess.CompletedProcess(
            args=command, stdout=docker_ps_output, returncode=0
        )

    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr(time, "sleep", lambda x: None)

    if expected_exception:
        with pytest.raises(expected_exception):
            server_manager.start_docker()
    else:
        server_manager.start_docker()


@pytest.mark.parametrize(
    "scenario, returncode, stdout, should_raise",
    [
        ("success", 0, "PostgreSQL is running and accepting connections", False),
        ("failure", 1, "PostgreSQL is not responding", True),
        ("exception", None, None, True),
    ],
)
def test_check_postgres(
    server_manager, monkeypatch, scenario, returncode, stdout, should_raise
):
    def mock_run(*args, **kwargs):
        if scenario == "exception":
            raise subprocess.CalledProcessError(1, args[0])

        class MockResult:
            def __init__(self):
                self.returncode = None
                self.stdout = ""
                self.stderr = ""

        result = MockResult()
        result.returncode = returncode
        result.stdout = stdout
        result.stderr = ""
        return result

    monkeypatch.setattr(subprocess, "run", mock_run)

    if should_raise:
        with pytest.raises(PostgresError):
            server_manager.check_postgres()
    else:
        assert server_manager.check_postgres() == True


@pytest.mark.parametrize(
    "scenario, isalembic, returncode, stdout, should_raise",
    [
        ("alembic_not_found", 0, False, None, True),
        ("failure", True, 1, "PostgreSQL is not responding", True),
        ("exception", True, None, None, True),
        ("success", True, 0, "PostgreSQL is running and accepting connections", False),
    ],
)
def test_run_migrations(
    server_manager, monkeypatch, scenario, isalembic, returncode, stdout, should_raise
):

    def mock_join(*args):
        if not isalembic:
            raise FileNotFoundError
        return "/".join(args)

    monkeypatch.setattr(os.path, "join", mock_join)

    def mock_run(*args, **kwargs):
        if scenario == "exception":
            raise subprocess.CalledProcessError(1, args[0])

        if returncode is None:
            return None

        class MockResult:
            def __init__(self, returncode: str):
                self.returncode = returncode
                self.stdout = ""
                self.stderr = ""

        result = MockResult(returncode=returncode)
        result.stdout = stdout
        result.stderr = ""
        return result

    monkeypatch.setattr(subprocess, "run", mock_run)

    if should_raise:
        with pytest.raises(MigrationError):
            server_manager.run_migrations()
    else:
        assert server_manager.run_migrations() is None


@pytest.mark.parametrize(
    "scenario, expected_exception",
    [
        ("server_running", None),
        ("server_already_running", StartServerError),
        ("docker_failure", StartServerError),
        ("postgres_failure", StartServerError),
        ("migration_failure", StartServerError),
    ],
)
def test_start_server(server_manager, monkeypatch, scenario, expected_exception):

    # Mock necessary methods
    monkeypatch.setattr(
        ServerManager,
        "start_docker",
        lambda self: (
            None
            if scenario != "docker_failure"
            else (_ for _ in ()).throw(DockerError("Docker failed"))
        ),
    )
    monkeypatch.setattr(
        ServerManager,
        "check_postgres",
        lambda self: (
            True
            if scenario != "postgres_failure"
            else (_ for _ in ()).throw(PostgresError("Postgres failed"))
        ),
    )
    monkeypatch.setattr(
        ServerManager,
        "run_migrations",
        lambda self: (
            None
            if scenario != "migration_failure"
            else (_ for _ in ()).throw(MigrationError("Migration failed"))
        ),
    )

    monkeypatch.setattr(
        os.path, "exists", lambda path: scenario == "server_already_running"
    )

    class MockProcess:
        def __init__(self):
            self.pid = 1234
            self.returncode = 0

        def poll(self):
            return True

        def wait(self):
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: MockProcess())

    def mock_open(*args, **kwargs):
        class MockFile:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def write(self, _):
                pass

        return MockFile()

    monkeypatch.setattr("builtins.open", mock_open)

    monkeypatch.setattr(time, "sleep", lambda x: None)

    if expected_exception:
        with pytest.raises(expected_exception):
            server_manager.start_server()
    else:
        try:
            server_manager.start_server()
        except StartServerError:
            pytest.fail("start_server raised StartServerError unexpectedly")


@pytest.mark.parametrize(
    "scenario, expected_exception",
    [
        ("server_not_running", StopServerError),  # PID file missing
        (
            "process_termination_failure",
            StopServerError,
        ),  # Process cannot be terminated
        ("docker_stop_failure", StopServerError),  # Docker fails to stop
        ("successful_shutdown", None),  # Everything works fine
    ],
)
def test_stop_server(server_manager, monkeypatch, scenario, expected_exception):

    pid_file_content = "1234\n5678"

    monkeypatch.setattr(
        os.path, "exists", lambda path: scenario != "server_not_running"
    )

    def mock_open(*args, **kwargs):
        class MockFile:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def read(self):
                return pid_file_content

        return MockFile()

    monkeypatch.setattr("builtins.open", mock_open)

    def mock_kill(pid, sig):
        if scenario == "process_termination_failure":
            raise StartServerError

    monkeypatch.setattr(os, "kill", mock_kill)

    def mock_subprocess_run(*args, **kwargs):
        if scenario == "docker_stop_failure":
            raise subprocess.CalledProcessError(1, "docker compose down")

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    monkeypatch.setattr(os, "remove", lambda path: None)

    if expected_exception != None:
        with pytest.raises(expected_exception):
            server_manager.stop_server()
    else:
        try:
            server_manager.stop_server()
        except StopServerError:
            pytest.fail("stop_server raised StopServerError unexpectedly")
