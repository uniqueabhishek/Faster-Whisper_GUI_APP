# Resume and Memory Management Features

This document describes the new automatic cleanup and resume capabilities added to the Faster-Whisper GUI application.

## Features Overview

### 1. Automatic Memory Cleanup

The application now automatically manages memory to prevent out-of-memory errors during long transcription sessions.

#### Memory Cleanup Strategies

- **Between Files**: Lightweight garbage collection after each file completes
- **Between Batches**: Aggressive memory cleanup after entire batch completes
- **After Errors**: Extra cleanup when memory errors occur
- **CUDA Cache**: Clears GPU memory cache (if using GPU)

#### Memory Monitoring

The application logs memory usage at key points:
- Before batch starts
- Before/after each file transcription
- After batch completes

Example log output:
```
[12:34:56] Batch start: Memory: 1250 MB (RSS), 2100 MB (VMS), 15.2%
[12:35:10] [File: audio.mp3] Starting - Memory: 1300 MB
[12:36:45] [File: audio.mp3] Finished - Memory: 1450 MB (Δ +150 MB)
[12:36:45] After cleanup: Memory: 1280 MB (RSS), 2050 MB (VMS), 14.8%
```

### 2. Session State Management

All transcription batches are now tracked in session files, allowing you to resume interrupted work.

#### Session Storage

- Location: `%TEMP%\faster_whisper_sessions\`
- Format: JSON files named by timestamp (e.g., `20251214_123456.json`)
- Contents: File list, status, errors, configuration

#### What Gets Saved

Each session tracks:
- All input files with their status (pending/processing/completed/failed)
- Model configuration (path, beam size, language, etc.)
- Output directory
- Temporary files created (for cleanup)
- Error messages for failed files
- Processing timestamps

### 3. Automatic Resume on Startup

When you launch the application, it automatically:

1. **Checks for incomplete sessions** from previous runs
2. **Shows resume dialog** if found, with:
   - Session details (date, progress, file count)
   - List of pending files
   - List of failed files (with error messages)
3. **Lets you choose**:
   - **Resume Session**: Continue from where you left off
   - **Discard & Start Fresh**: Delete the old session

#### Resume Dialog Example

```
┌─────────────────────────────────────────────┐
│ Previous transcription session found!       │
├─────────────────────────────────────────────┤
│ Session ID: 20251214_101530                 │
│ Created: 2025-12-14 10:15:30                │
│ Model: faster-whisper-medium                │
│ Total Files: 25                             │
│ Completed: 18                               │
│ Failed: 2                                   │
│ Pending: 5                                  │
│ Progress: 80%                               │
├─────────────────────────────────────────────┤
│ Pending Files:                              │
│   • file19.mp3                              │
│   • file20.mp3                              │
│   • ...                                     │
│                                             │
│ Failed Files (will retry):                  │
│   • file15.mp3 - Out of memory              │
│   • file16.mp3 - MemoryError                │
├─────────────────────────────────────────────┤
│ [Discard & Start Fresh] [Resume Session]   │
└─────────────────────────────────────────────┘
```

### 4. Temporary File Cleanup

The application now tracks and cleans up all temporary files.

#### Automatic Cleanup

- **During Processing**: Temp WAV files deleted immediately after transcription
- **On Completion**: All tracked temp files cleaned up
- **On Cancellation**: Partial files removed
- **On Startup**: Orphaned temp files older than 1 hour are deleted

#### Orphaned File Detection

The session manager scans `%TEMP%` on startup and removes:
- `.wav` files starting with `tmp` that are older than 1 hour
- These are likely from crashed or interrupted sessions

### 5. Enhanced Error Handling

#### Memory Error Detection

When `MemoryError` occurs:
1. Error is logged with file details
2. File is marked as "Failed (Memory Error)" in session
3. Aggressive memory cleanup is triggered
4. Processing continues with next file
5. Session is saved for potential resume

#### Error Recovery

Failed files can be retried by:
1. Resuming the session on next launch
2. The app will attempt to process failed files again
3. With memory cleanup between files, previously failed files may succeed

## Usage Guide

### Normal Workflow

1. **Add files** to queue as usual
2. **Click Start Transcription**
3. Files are processed with automatic memory cleanup
4. Session is tracked in background
5. On completion, session is automatically deleted (if all succeeded)

### Resume Workflow

1. **Application crashed** or you closed it during transcription
2. **Restart the application**
3. **Resume dialog appears** automatically
4. **Choose Resume Session**
5. Queue is populated with pending/failed files
6. **Click Start Transcription** to continue

### Manual Session Management

Sessions are stored in: `%TEMP%\faster_whisper_sessions\`

You can manually:
- View session files (JSON format)
- Delete old sessions
- Backup important sessions

## Configuration

### Memory Management Settings

Memory management happens automatically, but you can influence it:

- **Use smaller models** (tiny, base, small) for less memory usage
- **Reduce beam size** (use Fast Analysis instead of Deep Analysis)
- **Process fewer files** at once if memory is limited
- **Use int8** precision instead of float32

### Session Cleanup Behavior

Sessions are automatically cleaned up when:
- All files complete successfully
- You choose "Discard & Start Fresh" in resume dialog

Sessions are kept when:
- Any files fail during processing
- Application crashes or is closed during processing
- You cancel a batch

## Technical Details

### New Files Added

1. **session_manager.py** - Session state persistence
2. **memory_manager.py** - Memory monitoring and cleanup utilities
3. **resume_dialog.py** - Resume session UI dialog

### Modified Files

1. **workers.py** - Integrated session management and memory cleanup
2. **gui.py** - Added resume dialog on startup
3. **requirements.txt** - Added `psutil` dependency

### Session File Format

```json
{
  "session_id": "20251214_123456",
  "created_at": "2025-12-14T12:34:56",
  "model_path": "D:\\WhisperModel\\model\\faster-whisper-medium",
  "output_dir": "D:\\Output",
  "beam_size": 5,
  "vad_filter": false,
  "language": "en",
  "initial_prompt": null,
  "task": "transcribe",
  "patience": 1.0,
  "add_timestamps": true,
  "add_report": true,
  "files": [
    {
      "path": "D:\\Audio\\file1.mp3",
      "status": "completed",
      "error": null,
      "output_path": "D:\\Output\\file1.txt",
      "started_at": "2025-12-14T12:35:00",
      "completed_at": "2025-12-14T12:37:30"
    },
    {
      "path": "D:\\Audio\\file2.mp3",
      "status": "failed",
      "error": "MemoryError",
      "output_path": null,
      "started_at": "2025-12-14T12:37:31",
      "completed_at": "2025-12-14T12:38:45"
    },
    {
      "path": "D:\\Audio\\file3.mp3",
      "status": "pending",
      "error": null,
      "output_path": null,
      "started_at": null,
      "completed_at": null
    }
  ],
  "temp_files": []
}
```

## Benefits

### For Users

- ✅ **Never lose progress** - Resume from where you left off
- ✅ **Fewer crashes** - Automatic memory management prevents OOM errors
- ✅ **Cleaner system** - No orphaned temp files accumulating
- ✅ **Better reliability** - Failed files tracked and can be retried
- ✅ **Transparent** - See exactly what succeeded/failed

### For Long Batches

- ✅ **Process hundreds of files** without memory buildup
- ✅ **Recover from crashes** easily
- ✅ **Retry failed files** without reprocessing successes
- ✅ **Monitor memory usage** via logs

## Troubleshooting

### "Out of Memory" Errors Still Occurring

1. Use a smaller model (tiny, base, small instead of medium/large)
2. Use Fast Analysis (int8) instead of Precise/Deep Analysis
3. Process fewer files at once
4. Close other memory-intensive applications
5. Check available RAM in Task Manager

### Resume Dialog Not Appearing

1. Check `%TEMP%\faster_whisper_sessions\` for session files
2. Ensure previous batch had some files not completed
3. Successfully completed batches auto-delete their sessions

### Session Files Accumulating

Sessions are kept only when needed. To clean up:
1. Delete files from `%TEMP%\faster_whisper_sessions\`
2. Or let the app handle it (successful batches auto-delete)

### Memory Usage Still High

Memory cleanup is gradual. Python's garbage collector may not release memory immediately to the OS. The memory is freed within the process and will be reused for subsequent files.

## Future Enhancements

Potential improvements for future versions:

- Manual session browser/manager in GUI
- Configurable memory limits and alerts
- Automatic retry with lower beam size on memory errors
- Progress statistics and estimates based on session history
- Export session logs for analysis