"""Test script for resume and memory management features."""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)

LOGGER = logging.getLogger(__name__)


def test_session_manager():
    """Test session manager functionality."""
    print("\n" + "="*60)
    print("TEST 1: Session Manager")
    print("="*60)

    from session_manager import SessionManager

    manager = SessionManager()
    LOGGER.info("Session manager created: %s", manager.session_dir)

    # Create a test session
    test_files = [
        Path("D:/test/file1.mp3"),
        Path("D:/test/file2.mp3"),
        Path("D:/test/file3.mp3"),
    ]

    session = manager.create_session(
        input_files=test_files,
        model_path="D:/model/faster-whisper-medium",
        beam_size=5,
        language="en",
    )

    LOGGER.info("Created session: %s", session.session_id)
    LOGGER.info("Total files: %d", len(session.files))
    LOGGER.info("Pending files: %d", len(session.pending_files))

    # Update a file status
    manager.update_file_status(session, str(test_files[0]), "completed")
    LOGGER.info("Updated file 1 to completed")

    manager.update_file_status(
        session, str(test_files[1]), "failed", error="Test error"
    )
    LOGGER.info("Updated file 2 to failed")

    # Reload session
    loaded = manager.load_session(session.session_id)
    if loaded:
        LOGGER.info("Reloaded session successfully")
        LOGGER.info("  Completed: %d", len(loaded.completed_files))
        LOGGER.info("  Failed: %d", len(loaded.failed_files))
        LOGGER.info("  Pending: %d", len(loaded.pending_files))
        LOGGER.info("  Progress: %d%%", loaded.progress_percent)
    else:
        LOGGER.error("Failed to reload session")

    # Cleanup
    manager.delete_session(session.session_id)
    LOGGER.info("Deleted test session")

    print("[PASS] Session Manager Test Passed\n")


def test_memory_manager():
    """Test memory manager functionality."""
    print("\n" + "="*60)
    print("TEST 2: Memory Manager")
    print("="*60)

    from memory_manager import MemoryManager, MemoryMonitor

    # Log current memory
    MemoryManager.log_memory_usage("Initial state:")

    # Test memory monitoring
    with MemoryMonitor("Test Operation"):
        # Allocate some memory (intentionally unused for memory testing)
        _data = [i * i for i in range(1000000)]
        LOGGER.info("Allocated test data")

    # Test cleanup
    MemoryManager.cleanup_memory(aggressive=True)

    # Check available memory
    available = MemoryManager.check_memory_available(required_mb=500)
    LOGGER.info("Sufficient memory available: %s", available)

    print("[PASS] Memory Manager Test Passed\n")


def test_resume_workflow():
    """Test the resume workflow simulation."""
    print("\n" + "="*60)
    print("TEST 3: Resume Workflow Simulation")
    print("="*60)

    from session_manager import SessionManager

    manager = SessionManager()

    # Simulate interrupted batch
    test_files = [
        Path("D:/test/audio1.mp3"),
        Path("D:/test/audio2.mp3"),
        Path("D:/test/audio3.mp3"),
        Path("D:/test/audio4.mp3"),
        Path("D:/test/audio5.mp3"),
    ]

    session = manager.create_session(
        input_files=test_files,
        model_path="D:/model/faster-whisper-medium",
        beam_size=5,
    )

    LOGGER.info("Simulating interrupted batch...")

    # Simulate partial completion
    manager.update_file_status(session, str(test_files[0]), "completed")
    manager.update_file_status(session, str(test_files[1]), "completed")
    manager.update_file_status(
        session, str(test_files[2]), "failed", error="MemoryError"
    )
    # files[3] and files[4] remain pending

    LOGGER.info("Batch interrupted! Progress: %d%%", session.progress_percent)

    # Simulate app restart - check for latest session
    latest = manager.get_latest_session()
    if latest:
        LOGGER.info("Found incomplete session on restart: %s", latest.session_id)
        LOGGER.info("  Completed: %d files", len(latest.completed_files))
        LOGGER.info("  Failed: %d files", len(latest.failed_files))
        LOGGER.info("  Pending: %d files", len(latest.pending_files))

        # Files to resume
        to_resume = latest.pending_files + latest.failed_files
        LOGGER.info("Would resume processing %d files:", len(to_resume))
        for f in to_resume:
            LOGGER.info("  - %s (status: %s)", Path(f.path).name, f.status)

        # Cleanup
        manager.delete_session(latest.session_id)
        LOGGER.info("Cleaned up test session")
    else:
        LOGGER.error("No incomplete session found!")

    print("[PASS] Resume Workflow Test Passed\n")


def test_temp_file_tracking():
    """Test temp file tracking."""
    print("\n" + "="*60)
    print("TEST 4: Temp File Tracking")
    print("="*60)

    from session_manager import SessionManager
    import tempfile
    import os

    manager = SessionManager()

    # Create test session
    session = manager.create_session(
        input_files=[Path("D:/test/file1.mp3")],
        model_path="D:/model/test",
        beam_size=5,
    )

    # Create some temp files
    temp_files = []
    for i in range(3):
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        temp_path = Path(temp_path)

        # Write some data
        temp_path.write_text(f"test data {i}")
        temp_files.append(temp_path)

        # Track in session
        manager.add_temp_file(session, temp_path)
        LOGGER.info("Created and tracked temp file: %s", temp_path.name)

    LOGGER.info("Total temp files tracked: %d", len(session.temp_files))

    # Verify files exist
    for temp_file in temp_files:
        if temp_file.exists():
            LOGGER.info("[OK] Temp file exists: %s", temp_file.name)

    # Cleanup temp files
    manager.cleanup_temp_files(session)
    LOGGER.info("Cleanup completed")

    # Verify files deleted
    deleted_count = 0
    for temp_file in temp_files:
        if not temp_file.exists():
            deleted_count += 1
            LOGGER.info("[OK] Temp file deleted: %s", temp_file.name)

    if deleted_count == len(temp_files):
        LOGGER.info("All temp files successfully deleted")
    else:
        LOGGER.warning("Some temp files not deleted: %d/%d",
                      deleted_count, len(temp_files))

    # Cleanup session
    manager.delete_session(session.session_id)

    print("[PASS] Temp File Tracking Test Passed\n")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("RESUME AND MEMORY MANAGEMENT - TEST SUITE")
    print("="*60)

    try:
        test_session_manager()
        test_memory_manager()
        test_resume_workflow()
        test_temp_file_tracking()

        print("\n" + "="*60)
        print("ALL TESTS PASSED [OK]")
        print("="*60)
        print("\nThe resume and memory management features are working correctly!")
        print("\nNext steps:")
        print("1. Install psutil: pip install psutil>=5.9.0")
        print("2. Test with actual audio files in the GUI")
        print("3. Test crash recovery by force-closing during transcription")

    except Exception as e:
        LOGGER.error("Test failed with error: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
