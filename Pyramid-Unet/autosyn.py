import os
import time
import logging
import threading
import paramiko
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Operation Type Enumeration
class OperationType(Enum):
    UPLOAD = "upload"
    DOWNLOAD = "download"
    DELETE_LOCAL = "delete_local"
    DELETE_REMOTE = "delete_remote"


# Configuration Class
@dataclass
class Config:
    host: str = "10.254.254.1"
    port: int = 22
    username: str = "jpkuser"
    password: str = "jpkjpk"  # Password hardcoded in configuration
    local_dir: str = r"C:\Users\17105\Desktop\data-transmit"
    remote_dir: str = "/home/jpkuser/Desktop/data-transmit"
    log_file: str = r"C:\sync_logs\sftp_sync.log"
    check_interval: int = 10  # Interval (seconds) for checking remote changes
    max_retries: int = 3  # Maximum retries for failed operations
    excluded_patterns: List[str] = field(default_factory=lambda: [
        '.git', 'node_modules', '*.tmp', '*.log', 'logs/'
    ])


# File Status Class
@dataclass
class FileState:
    mtime: float
    size: int
    exists: bool
    last_checked: float = field(default_factory=time.time)


# SFTP Connection Manager
class SftpConnector:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.ssh = None
        self.sftp = None
        self.connected = False

    def connect(self) -> bool:
        """Establish SFTP connection"""
        try:
            if self.connected:
                self.disconnect()

            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(
                self.config.host,
                port=self.config.port,
                username=self.config.username,
                password=self.config.password,
                timeout=10
            )

            self.sftp = self.ssh.open_sftp()
            self.connected = True
            self.logger.info("SFTP connection established successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to establish SFTP connection: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect SFTP connection"""
        try:
            if self.sftp:
                self.sftp.close()
            if self.ssh:
                self.ssh.close()
            self.connected = False
            self.logger.info("SFTP connection disconnected")
        except Exception as e:
            self.logger.error(f"Error disconnecting SFTP connection: {e}")

    def execute_with_retry(self, func, *args, **kwargs):
        """SFTP operation executor with retry mechanism"""
        retries = 0
        while retries < self.config.max_retries:
            try:
                if not self.connected and not self.connect():
                    retries += 1
                    time.sleep(2)
                    continue
                return func(*args, **kwargs)
            except FileNotFoundError as e:
                self.logger.warning(f"File not found: {e}")
                self.connected = False
                return False  # File not found, no need to retry
            except Exception as e:
                self.logger.error(f"SFTP operation failed: {e}")
                self.connected = False
                retries += 1
                time.sleep(2)
        self.logger.error(f"Operation reached maximum retries: {func.__name__}")
        return None

    def file_exists(self, remote_path: str) -> bool:
        """Check if remote file exists"""

        def _check():
            try:
                self.sftp.stat(remote_path)
                return True
            except FileNotFoundError:
                return False

        return self.execute_with_retry(_check)

    def get_remote_file_list(self) -> Optional[Dict[str, Dict]]:
        """Get list of files in remote directory"""

        def _list_files():
            files = {}
            try:
                for item in self.sftp.listdir_attr(self.config.remote_dir):
                    if item.filename in ['.', '..']:
                        continue
                    files[item.filename] = {
                        'size': item.st_size,
                        'mtime': item.st_mtime,
                        'is_dir': item.st_mode & 0o40000 != 0
                    }
            except Exception as e:
                self.logger.error(f"Failed to get remote file list: {e}")
                return None
            return files

        return self.execute_with_retry(_list_files)

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload file to remote"""

        def _upload():
            # Ensure target directory exists
            remote_dir = os.path.dirname(remote_path)
            self._ensure_remote_dir_exists(remote_dir)

            # Upload file
            self.sftp.put(local_path, remote_path)
            return True

        return self.execute_with_retry(_upload)

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download file from remote"""

        def _download():
            # Check if remote file exists
            if not self.file_exists(remote_path):
                self.logger.warning(f"Remote file does not exist, skipping download: {remote_path}")
                return False

            # Ensure local directory exists
            local_dir = os.path.dirname(local_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            # Download file
            self.sftp.get(remote_path, local_path)
            return True

        return self.execute_with_retry(_download)

    def delete_remote_file(self, remote_path: str) -> bool:
        """Delete remote file"""

        def _delete():
            # Check if remote file exists
            if not self.file_exists(remote_path):
                self.logger.warning(f"Remote file does not exist, skipping deletion: {remote_path}")
                return True

            # Delete file
            self.sftp.remove(remote_path)
            return True

        return self.execute_with_retry(_delete)

    def _ensure_remote_dir_exists(self, remote_dir: str):
        """Ensure remote directory exists"""
        try:
            self.sftp.stat(remote_dir)
        except FileNotFoundError:
            parent_dir = os.path.dirname(remote_dir)
            if parent_dir != remote_dir:
                self._ensure_remote_dir_exists(parent_dir)
            self.sftp.mkdir(remote_dir)


# Local File System Event Handler
class LocalSyncHandler(FileSystemEventHandler):
    def __init__(self, sync_manager):
        self.sync_manager = sync_manager

    def on_any_event(self, event):
        """Handle file system events"""
        if event.is_directory or self.sync_manager.is_excluded(event.src_path):
            return

        # Brief delay to avoid incomplete file operations
        time.sleep(0.1)

        # Calculate relative path and remote path
        rel_path = os.path.relpath(event.src_path, self.sync_manager.config.local_dir)
        remote_path = os.path.join(self.sync_manager.config.remote_dir, rel_path).replace('\\', '/')

        if event.event_type == 'created' or event.event_type == 'modified':
            self.sync_manager.logger.info(f"Local creation/modification detected: {rel_path}")
            self.sync_manager.perform_sync(OperationType.UPLOAD, event.src_path, remote_path)

        elif event.event_type == 'deleted':
            self.sync_manager.logger.info(f"Local deletion detected: {rel_path}")
            self.sync_manager.perform_sync(OperationType.DELETE_REMOTE, None, remote_path)


# File Synchronization Manager
class SyncManager:
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        self.sftp = SftpConnector(config, self.logger)
        self.running = False
        self.observer = None
        self.remote_watcher_thread = None
        self.sync_lock = threading.Lock()  # Synchronization lock
        self.local_state: Dict[str, FileState] = {}  # Local file state
        self.remote_state: Dict[str, FileState] = {}  # Remote file state

    def _setup_logger(self):
        """Setup logger"""
        logger = logging.getLogger("sftp_sync")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # Ensure log directory exists
            log_dir = os.path.dirname(self.config.log_file)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            file_handler = logging.FileHandler(self.config.log_file, encoding='utf-8')
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger

    def is_excluded(self, path: str) -> bool:
        """Check if path should be excluded"""
        for pattern in self.config.excluded_patterns:
            if pattern.startswith('*.') and path.endswith(pattern[1:]):
                return True
            if pattern in path:
                return True
        return False

    def start(self):
        """Start synchronization service"""
        self.logger.info("Starting file synchronization service...")
        self.running = True

        self.initial_sync()
        self._start_local_watcher()
        self._start_remote_watcher()

        self.logger.info("File synchronization service started")

    def stop(self):
        """Stop synchronization service"""
        self.logger.info("Stopping file synchronization service...")
        self.running = False

        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)

        if self.remote_watcher_thread:
            self.remote_watcher_thread.join(timeout=5)

        self.sftp.disconnect()
        self.logger.info("File synchronization service stopped")

    def perform_sync(self, action: OperationType, local_path: Optional[str], remote_path: Optional[str]):
        """Perform synchronization operation"""
        self.logger.info(f"Performing synchronization operation: {action.value} {local_path or remote_path}")

        with self.sync_lock:  # Ensure only one synchronization operation at a time
            success = False
            retries = 0

            while retries < self.config.max_retries:
                try:
                    if action == OperationType.UPLOAD:
                        if not os.path.exists(local_path):
                            self.logger.warning(f"Local file does not exist, skipping upload: {local_path}")
                            return False

                        success = self.sftp.upload_file(local_path, remote_path)
                        if success:
                            # Update local and remote states
                            rel_path = os.path.relpath(local_path, self.config.local_dir)
                            try:
                                stat = os.stat(local_path)
                                self.local_state[rel_path] = FileState(
                                    mtime=stat.st_mtime,
                                    size=stat.st_size,
                                    exists=True,
                                    last_checked=time.time()
                                )
                                self.remote_state[rel_path] = FileState(
                                    mtime=stat.st_mtime,
                                    size=stat.st_size,
                                    exists=True,
                                    last_checked=time.time()
                                )
                            except Exception as e:
                                self.logger.warning(f"Failed to update file state: {e}")

                    elif action == OperationType.DOWNLOAD:
                        success = self.sftp.download_file(remote_path, local_path)
                        if success:
                            # Update local and remote states
                            rel_path = os.path.relpath(local_path, self.config.local_dir)
                            try:
                                stat = os.stat(local_path)
                                self.local_state[rel_path] = FileState(
                                    mtime=stat.st_mtime,
                                    size=stat.st_size,
                                    exists=True,
                                    last_checked=time.time()
                                )
                                self.remote_state[rel_path] = FileState(
                                    mtime=stat.st_mtime,
                                    size=stat.st_size,
                                    exists=True,
                                    last_checked=time.time()
                                )
                            except Exception as e:
                                self.logger.warning(f"Failed to update file state: {e}")

                    elif action == OperationType.DELETE_REMOTE:
                        # Calculate relative path
                        rel_path = os.path.basename(remote_path) if remote_path else ""

                        # Check if remote file exists
                        if self.sftp.file_exists(remote_path):
                            success = self.sftp.delete_remote_file(remote_path)
                            if success:
                                # Update remote state immediately to prevent duplicate operations
                                self.remote_state[rel_path] = FileState(
                                    mtime=0,
                                    size=0,
                                    exists=False,
                                    last_checked=time.time()
                                )
                        else:
                            success = True  # File does not exist, consider operation successful

                    elif action == OperationType.DELETE_LOCAL:
                        if os.path.exists(local_path):
                            try:
                                # Attempt to delete file
                                os.remove(local_path)
                                success = True
                                # Update local state
                                rel_path = os.path.relpath(local_path, self.config.local_dir)
                                self.local_state[rel_path] = FileState(
                                    mtime=0,
                                    size=0,
                                    exists=False,
                                    last_checked=time.time()
                                )
                            except PermissionError as e:
                                self.logger.warning(f"File is locked, cannot delete: {local_path}, Error: {e}")
                                success = False
                        else:
                            success = True  # File does not exist, consider operation successful

                    if success:
                        self.logger.info(f"Synchronization successful: {action.value} {local_path or remote_path}")
                        break
                    else:
                        retries += 1
                        self.logger.warning(f"Synchronization failed, retrying ({retries}/{self.config.max_retries}): {action.value}")
                        time.sleep(1)

                except Exception as e:
                    retries += 1
                    self.logger.error(f"Synchronization error, retrying ({retries}/{self.config.max_retries}): {e}")
                    time.sleep(1)

            if not success:
                self.logger.error(f"Final synchronization failure: {action.value} {local_path or remote_path}")

    def initial_sync(self):
        """Perform initial two-way synchronization"""
        self.logger.info("Performing initial two-way synchronization...")

        local_files = self._get_local_file_list()
        remote_files = self.sftp.get_remote_file_list()

        if not local_files or not remote_files:
            self.logger.warning("Initial synchronization failed: Cannot retrieve file lists")
            return

        # Compare files on both sides, use latest modification time as criterion
        for filename in set(local_files.keys()).union(set(remote_files.keys())):
            local_info = local_files.get(filename)
            remote_info = remote_files.get(filename)

            if local_info and not remote_info:
                # Local file exists, remote does not -> Upload
                local_path = os.path.join(self.config.local_dir, filename)
                remote_path = os.path.join(self.config.remote_dir, filename).replace('\\', '/')
                self.perform_sync(OperationType.UPLOAD, local_path, remote_path)

            elif not local_info and remote_info:
                # Local file does not exist, remote does -> Download
                local_path = os.path.join(self.config.local_dir, filename)
                remote_path = os.path.join(self.config.remote_dir, filename).replace('\\', '/')
                self.perform_sync(OperationType.DOWNLOAD, local_path, remote_path)

            elif local_info and remote_info:
                # Both exist, compare modification times
                if local_info['mtime'] > remote_info['mtime']:
                    # Local is newer -> Upload
                    local_path = os.path.join(self.config.local_dir, filename)
                    remote_path = os.path.join(self.config.remote_dir, filename).replace('\\', '/')
                    self.perform_sync(OperationType.UPLOAD, local_path, remote_path)
                elif remote_info['mtime'] > local_info['mtime']:
                    # Remote is newer -> Download
                    local_path = os.path.join(self.config.local_dir, filename)
                    remote_path = os.path.join(self.config.remote_dir, filename).replace('\\', '/')
                    self.perform_sync(OperationType.DOWNLOAD, local_path, remote_path)

        # Update states
        self._update_file_states(local_files, remote_files)

    def _update_file_states(self, local_files: Dict, remote_files: Dict):
        """Update file states"""
        current_time = time.time()

        # Update local state
        for filename, info in local_files.items():
            self.local_state[filename] = FileState(
                mtime=info['mtime'],
                size=info['size'],
                exists=True,
                last_checked=current_time
            )

        # Handle deleted local files
        for filename in list(self.local_state.keys()):
            if filename not in local_files:
                if self.local_state[filename].exists:
                    self.local_state[filename] = FileState(
                        mtime=0,
                        size=0,
                        exists=False,
                        last_checked=current_time
                    )
                else:
                    # If already marked as non-existent and not checked for a long time, remove state
                    if current_time - self.local_state[filename].last_checked > 3600:
                        del self.local_state[filename]

        # Update remote state
        for filename, info in remote_files.items():
            self.remote_state[filename] = FileState(
                mtime=info['mtime'],
                size=info['size'],
                exists=True,
                last_checked=current_time
            )

        # Handle deleted remote files
        for filename in list(self.remote_state.keys()):
            if filename not in remote_files:
                if self.remote_state[filename].exists:
                    self.remote_state[filename] = FileState(
                        mtime=0,
                        size=0,
                        exists=False,
                        last_checked=current_time
                    )
                else:
                    # If already marked as non-existent and not checked for a long time, remove state
                    if current_time - self.remote_state[filename].last_checked > 3600:
                        del self.remote_state[filename]

    def _get_local_file_list(self) -> Dict[str, Dict]:
        """Get list of local directory files"""
        files = {}
        try:
            for item in os.listdir(self.config.local_dir):
                item_path = os.path.join(self.config.local_dir, item)
                if self.is_excluded(item_path):
                    continue
                try:
                    stat = os.stat(item_path)
                    files[item] = {
                        'size': stat.st_size,
                        'mtime': stat.st_mtime,
                        'is_dir': os.path.isdir(item_path)
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to retrieve file information: {item_path}, Error: {e}")
            return files
        except Exception as e:
            self.logger.error(f"Failed to get local file list: {e}")
            return {}

    def _start_local_watcher(self):
        """Start local file monitoring"""
        event_handler = LocalSyncHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, self.config.local_dir, recursive=True)
        self.observer.start()
        self.logger.info(f"Started monitoring local directory: {self.config.local_dir}")

    def _start_remote_watcher(self):
        """Start remote file monitoring thread"""
        self.remote_watcher_thread = threading.Thread(target=self._watch_remote_changes, daemon=True)
        self.remote_watcher_thread.start()
        self.logger.info("Started monitoring remote directory")

    def _watch_remote_changes(self):
        """Monitor remote file system changes"""
        while self.running:
            try:
                # Get current remote file list
                remote_files = self.sftp.get_remote_file_list()
                if not remote_files:
                    time.sleep(self.config.check_interval)
                    continue

                # Compare with previous state
                for filename in set(remote_files.keys()).union(set(self.remote_state.keys())):
                    remote_info = remote_files.get(filename)
                    remote_state = self.remote_state.get(filename)

                    # Remote file added
                    if remote_info and (not remote_state or not remote_state.exists):
                        self.logger.info(f"Remote addition detected: {filename}")
                        local_path = os.path.join(self.config.local_dir, filename)
                        remote_path = os.path.join(self.config.remote_dir, filename).replace('\\', '/')
                        self.perform_sync(OperationType.DOWNLOAD, local_path, remote_path)

                    # Remote file deleted
                    elif not remote_info and remote_state and remote_state.exists:
                        self.logger.info(f"Remote deletion detected: {filename}")
                        local_path = os.path.join(self.config.local_dir, filename)
                        # First check if local file exists
                        if os.path.exists(local_path):
                            self.perform_sync(OperationType.DELETE_LOCAL, local_path, None)
                        else:
                            # File already deleted, directly update state
                            rel_path = os.path.relpath(local_path, self.config.local_dir)
                            self.local_state[rel_path] = FileState(
                                mtime=0,
                                size=0,
                                exists=False,
                                last_checked=time.time()
                            )
                            self.logger.info(f"Local file already deleted, updating state directly: {local_path}")

                # Update remote state
                self._update_file_states({}, remote_files)

            except Exception as e:
                self.logger.error(f"Error monitoring remote changes: {e}")

            time.sleep(self.config.check_interval)


# Main program
def main():
    config = Config()

    # Validate configuration
    if not os.path.exists(config.local_dir):
        print(f"Error: Local directory does not exist - {config.local_dir}")
        return 1

    sync_manager = SyncManager(config)

    try:
        sync_manager.start()
        print("Press Ctrl+C to stop service...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        sync_manager.stop()
        return 0


if __name__ == "__main__":
    main()