import logging
from typing import Callable, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from text_anonymizer.config_cache import ConfigCache

logger = logging.getLogger(__name__)


class ConfigReloadHandler(FileSystemEventHandler):
    """Watch config directory and invalidate cache on file changes."""

    def __init__(self, on_change_callback: Optional[Callable[[], None]] = None):
        super().__init__()
        self._on_change_callback = on_change_callback

    def on_modified(self, event):
        if event.is_directory:
            return
        logger.debug("Config file changed: %s", event.src_path)
        ConfigCache.instance().notify_path_changed(event.src_path)
        if self._on_change_callback:
            self._on_change_callback()


class ConfigWatcher:
    """Filesystem watcher for config directory changes."""

    def __init__(
        self,
        config_dir: str,
        enabled: bool = True,
        on_change_callback: Optional[Callable[[], None]] = None
    ):
        self.config_dir = config_dir
        self.enabled = enabled
        self.observer: Optional[Observer] = None
        self._on_change_callback = on_change_callback

    def start(self) -> None:
        if not self.enabled:
            logger.info("ConfigWatcher is disabled")
            return
        event_handler = ConfigReloadHandler(on_change_callback=self._on_change_callback)
        self.observer = Observer()
        self.observer.schedule(event_handler, self.config_dir, recursive=True)
        self.observer.start()
        logger.info("ConfigWatcher started for %s (recursive)", self.config_dir)

    def stop(self) -> None:
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("ConfigWatcher stopped")

