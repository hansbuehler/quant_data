"""Persistent top-level failure reporting for FLEX command-line workers."""

from __future__ import annotations

import faulthandler
import json
import os
import signal
import socket
import sys
import tempfile
import threading
import traceback
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import NoReturn

from cdxcore.err import verify_inp
from cdxcore.util import is_filename

_DEFAULT_FAILURE_DIRECTORY = Path(__file__).with_name("reports") / "failures"
_DIRECTORY_OPTIONS = ("--report-directory", "--output-directory")
_FILE_OPTIONS = ("--output",)
GIB = 1024**3
RAM_WARNING_AVAILABLE_BYTES = 200 * GIB
RAM_TERMINATE_AVAILABLE_BYTES = 100 * GIB
RAM_POLL_SECONDS = 30.0


class LowAvailableMemoryError(RuntimeError):
    """Stop a FLEX worker before system-wide memory exhaustion."""

    def __init__(
        self,
        *,
        available_bytes: int,
        warning_bytes: int,
        terminate_bytes: int,
    ) -> None:
        self.available_bytes = available_bytes
        self.warning_bytes = warning_bytes
        self.terminate_bytes = terminate_bytes
        super().__init__(
            "FLEX RAM safety termination: "
            f"MemAvailable={available_bytes / GIB:.3f} GiB is at or below "
            f"the {terminate_bytes / GIB:.3f} GiB termination threshold"
        )


class _TerminationSignal(BaseException):
    """Convert a catchable process-termination signal into a persisted failure."""

    def __init__(self, signal_number: int) -> None:
        self.signal_number = signal_number
        try:
            self.signal_name = signal.Signals(signal_number).name
        except ValueError:
            self.signal_name = f"signal-{signal_number}"
        super().__init__(f"Process received {self.signal_name} ({signal_number})")


def available_memory_bytes(
    meminfo_path: Path = Path("/proc/meminfo"),
) -> int:
    """Read Linux reclaimable available memory from ``/proc/meminfo``."""
    for line in meminfo_path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition(":")
        if name != "MemAvailable" or separator == "":
            continue
        fields = value.split()
        if len(fields) != 2 or fields[1] != "kB":
            raise RuntimeError(f"Invalid MemAvailable row in {meminfo_path}: {line!r}")
        available = int(fields[0]) * 1024
        if available <= 0:
            raise RuntimeError(
                f"Non-positive MemAvailable value in {meminfo_path}: {available}"
            )
        return available
    raise RuntimeError(f"MemAvailable is missing from {meminfo_path}")


def failure_directory_from_argv(
    argv: Sequence[str] | None = None,
) -> Path:
    """Locate a CLI-specific failure directory without invoking its parser."""
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    for option in _DIRECTORY_OPTIONS:
        value = _option_value(arguments, option)
        if value is not None:
            return Path(value).expanduser() / "failures"
    for option in _FILE_OPTIONS:
        value = _option_value(arguments, option)
        if value is not None:
            return Path(value).expanduser().parent / "failures"
    return _DEFAULT_FAILURE_DIRECTORY


def _option_value(arguments: Sequence[str], option: str) -> str | None:
    prefix = f"{option}="
    for index, argument in enumerate(arguments):
        if argument.startswith(prefix):
            return argument[len(prefix) :]
        if argument == option and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


def _failure_identity(program: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{program}-{timestamp}-pid{os.getpid()}"


def _atomic_write_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _exception_payload(
    exception: BaseException,
    *,
    program: str,
    failure_id: str,
    source: str,
    thread_name: str,
    extra: Mapping[str, object] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "flex-process-failure-v1",
        "failure_id": failure_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "program": program,
        "source": source,
        "exception": {
            "module": type(exception).__module__,
            "type": type(exception).__qualname__,
            "message": str(exception),
            "traceback": "".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            ),
        },
        "process": {
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "thread": thread_name,
            "hostname": socket.gethostname(),
            "working_directory": str(Path.cwd()),
            "argv": list(sys.argv),
            "python": sys.version,
            "executable": sys.executable,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    if isinstance(exception, _TerminationSignal):
        payload["signal"] = {
            "number": exception.signal_number,
            "name": exception.signal_name,
        }
    if extra is not None:
        payload["extra"] = dict(extra)
    return payload


def record_exception(
    exception: BaseException,
    *,
    program: str,
    failure_directory: Path,
    source: str = "main",
    thread_name: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Atomically persist a unique exception record and a latest pointer."""
    verify_inp(
        isinstance(program, str) and program not in {"", ".", ".."} and is_filename(program),
        lambda: f"Expected a valid failure-report program name, found {program!r}",
    )
    failure_id = _failure_identity(program)
    payload = _exception_payload(
        exception,
        program=program,
        failure_id=failure_id,
        source=source,
        thread_name=threading.current_thread().name if thread_name is None else thread_name,
        extra=extra,
    )
    unique_path = failure_directory / f"{failure_id}.failure.json"
    latest_path = failure_directory / f"{program}-latest.failure.json"
    _atomic_write_json(payload, unique_path)
    _atomic_write_json(payload, latest_path)
    return unique_path


def _record_with_fallback(
    exception: BaseException,
    *,
    program: str,
    failure_directory: Path,
    source: str,
    thread_name: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> Path | None:
    directories = (failure_directory, _DEFAULT_FAILURE_DIRECTORY)
    errors: list[str] = []
    for directory in dict.fromkeys(directories):
        try:
            path = record_exception(
                exception,
                program=program,
                failure_directory=directory,
                source=source,
                thread_name=thread_name,
                extra=extra,
            )
            print(f"FLEX failure record: {path}", file=sys.stderr, flush=True)
            return path
        except BaseException as recording_error:
            errors.append(f"{directory}: {recording_error!r}")
    print(
        "FLEX could not persist its failure record: " + "; ".join(errors),
        file=sys.stderr,
        flush=True,
    )
    return None


def _termination_handler(
    signal_number: int,
    _frame: FrameType | None,
) -> NoReturn:
    raise _TerminationSignal(signal_number)


def run_guarded_cli(
    main: Callable[[], None],
    *,
    program: str,
    failure_directory: Path | None = None,
    clean_sigterm: bool = False,
    enable_ram_guard: bool = True,
    ram_warning_available_bytes: int = RAM_WARNING_AVAILABLE_BYTES,
    ram_terminate_available_bytes: int = RAM_TERMINATE_AVAILABLE_BYTES,
    ram_poll_seconds: float = RAM_POLL_SECONDS,
    ram_meminfo_path: Path = Path("/proc/meminfo"),
) -> None:
    """Run a CLI with durable reporting for uncaught and fatal failures.

    Python exceptions, catchable termination signals, uncaught background-thread
    exceptions, and unraisable exceptions receive structured JSON records. A
    temporary faulthandler log survives only a fatal interpreter/native crash.
    The original exception or nonzero signal exit is preserved. Long-lived
    supervisors may set ``clean_sigterm`` so an intentional ``systemctl stop``
    exits normally without manufacturing a failure record; RAM-guard SIGTERM
    remains a persisted failure. A supervisor which implements the project-wide
    RAM stop itself may disable this process-local guard.
    """
    verify_inp(callable(main), lambda: f"Expected a callable main, found {main!r}")
    verify_inp(
        isinstance(clean_sigterm, bool),
        lambda: f"'clean_sigterm' must be bool, found {clean_sigterm!r}",
    )
    verify_inp(
        isinstance(enable_ram_guard, bool),
        lambda: f"'enable_ram_guard' must be bool, found {enable_ram_guard!r}",
    )
    verify_inp(
        isinstance(ram_terminate_available_bytes, int)
        and ram_terminate_available_bytes > 0,
        lambda: "RAM termination threshold must be a positive integer number of bytes",
    )
    verify_inp(
        isinstance(ram_warning_available_bytes, int)
        and ram_warning_available_bytes > ram_terminate_available_bytes,
        lambda: "RAM warning threshold must exceed the termination threshold",
    )
    verify_inp(
        ram_poll_seconds > 0.0,
        lambda: "RAM polling interval must be positive",
    )
    directory = (
        failure_directory_from_argv()
        if failure_directory is None
        else Path(failure_directory)
    )
    directory.mkdir(parents=True, exist_ok=True)
    fatal_identity = _failure_identity(program)
    fatal_path = directory / f"{fatal_identity}.fatal.log"
    fatal_file = fatal_path.open("a", encoding="utf-8")
    faulthandler_was_enabled = faulthandler.is_enabled()
    faulthandler.enable(file=fatal_file, all_threads=True)

    previous_thread_hook = threading.excepthook
    previous_unraisable_hook = sys.unraisablehook
    previous_signal_handlers: dict[int, signal.Handlers] = {}
    ram_stop = threading.Event()
    ram_failures: list[LowAvailableMemoryError] = []
    ram_monitor: threading.Thread | None = None
    ram_warning_active = False

    def check_ram() -> None:
        nonlocal ram_warning_active
        available = available_memory_bytes(ram_meminfo_path)
        if available <= ram_terminate_available_bytes:
            raise LowAvailableMemoryError(
                available_bytes=available,
                warning_bytes=ram_warning_available_bytes,
                terminate_bytes=ram_terminate_available_bytes,
            )
        if available < ram_warning_available_bytes:
            if not ram_warning_active:
                print(
                    "FLEX RAM warning: "
                    f"MemAvailable={available / GIB:.3f} GiB is below "
                    f"{ram_warning_available_bytes / GIB:.3f} GiB",
                    file=sys.stderr,
                    flush=True,
                )
            ram_warning_active = True
        else:
            ram_warning_active = False

    def monitor_ram() -> None:
        while not ram_stop.wait(ram_poll_seconds):
            try:
                check_ram()
            except LowAvailableMemoryError as exception:
                ram_failures.append(exception)
                print(str(exception), file=sys.stderr, flush=True)
                os.kill(os.getpid(), signal.SIGTERM)
                return
            except BaseException as exception:
                # Continuing a long independent worker without the required
                # memory guard would silently violate the project safety rule.
                monitoring_failure = LowAvailableMemoryError(
                    available_bytes=0,
                    warning_bytes=ram_warning_available_bytes,
                    terminate_bytes=ram_terminate_available_bytes,
                )
                monitoring_failure.add_note(
                    f"RAM monitor could not read {ram_meminfo_path}: {exception!r}"
                )
                ram_failures.append(monitoring_failure)
                print(
                    f"FLEX RAM monitor failure: {exception!r}",
                    file=sys.stderr,
                    flush=True,
                )
                os.kill(os.getpid(), signal.SIGTERM)
                return

    def thread_hook(arguments: threading.ExceptHookArgs) -> None:
        _record_with_fallback(
            arguments.exc_value,
            program=program,
            failure_directory=directory,
            source="thread",
            thread_name=arguments.thread.name if arguments.thread is not None else "unknown",
        )
        previous_thread_hook(arguments)
        # Continuing after an unexpected training-thread death risks silently
        # producing a partial model. The fsynced report survives this hard exit.
        os._exit(1)

    def unraisable_hook(arguments: sys.UnraisableHookArgs) -> None:
        exception = arguments.exc_value or RuntimeError(
            f"Unraisable exception without value: {arguments.err_msg}"
        )
        _record_with_fallback(
            exception,
            program=program,
            failure_directory=directory,
            source="unraisable",
            extra={"err_msg": arguments.err_msg, "object": repr(arguments.object)},
        )
        previous_unraisable_hook(arguments)

    threading.excepthook = thread_hook
    sys.unraisablehook = unraisable_hook
    for signal_number in filter(
        None,
        (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)),
    ):
        previous_signal_handlers[signal_number] = signal.getsignal(signal_number)
        signal.signal(signal_number, _termination_handler)

    try:
        if enable_ram_guard:
            check_ram()
            ram_monitor = threading.Thread(
                target=monitor_ram,
                name="flex-ram-safety-monitor",
                daemon=True,
            )
            ram_monitor.start()
        main()
    except SystemExit as exception:
        if exception.code not in {None, 0}:
            _record_with_fallback(
                exception,
                program=program,
                failure_directory=directory,
                source="main",
            )
        raise
    except _TerminationSignal as exception:
        if ram_failures:
            ram_failure = ram_failures[-1]
            _record_with_fallback(
                ram_failure,
                program=program,
                failure_directory=directory,
                source="ram_guard",
                extra={
                    "available_bytes": ram_failure.available_bytes,
                    "warning_available_bytes": ram_failure.warning_bytes,
                    "terminate_available_bytes": ram_failure.terminate_bytes,
                    "meminfo_path": str(ram_meminfo_path),
                },
            )
        elif not (
            clean_sigterm
            and exception.signal_number == getattr(signal, "SIGTERM", None)
        ):
            _record_with_fallback(
                exception,
                program=program,
                failure_directory=directory,
                source="signal",
            )
        else:
            print("FLEX supervisor received an intentional SIGTERM", flush=True)
            return
        raise SystemExit(128 + exception.signal_number) from None
    except LowAvailableMemoryError as exception:
        _record_with_fallback(
            exception,
            program=program,
            failure_directory=directory,
            source="ram_guard",
            extra={
                "available_bytes": exception.available_bytes,
                "warning_available_bytes": exception.warning_bytes,
                "terminate_available_bytes": exception.terminate_bytes,
                "meminfo_path": str(ram_meminfo_path),
            },
        )
        raise
    except BaseException as exception:
        _record_with_fallback(
            exception,
            program=program,
            failure_directory=directory,
            source="main",
        )
        raise
    finally:
        ram_stop.set()
        if ram_monitor is not None and ram_monitor is not threading.current_thread():
            ram_monitor.join(timeout=min(ram_poll_seconds, 1.0))
        for signal_number, previous_handler in previous_signal_handlers.items():
            signal.signal(signal_number, previous_handler)
        threading.excepthook = previous_thread_hook
        sys.unraisablehook = previous_unraisable_hook
        faulthandler.disable()
        if faulthandler_was_enabled:
            faulthandler.enable()
        fatal_file.close()
        # A Python-level exception already has structured JSON. If the process
        # dies in native code, this finally block never runs and the fatal log
        # remains as the only available traceback.
        if fatal_path.exists():
            fatal_path.unlink()
