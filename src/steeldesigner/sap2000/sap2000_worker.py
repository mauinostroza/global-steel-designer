"""Worker QThread genérico para ejecutar llamadas SAP2000/COM fuera del hilo de UI.

Las llamadas COM a SAP2000 pueden ser lentas (conectar, leer resultados de
muchos frames, etc.). Ejecutarlas en el hilo principal bloquea la interfaz.
Este worker corre una función arbitraria en un QThread y reporta resultado
o error mediante señales Qt, siguiendo la regla de la guía de integración:
"las llamadas lentas a SAP2000 deben ejecutarse fuera del hilo principal".
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QThread, Signal


class Sap2000Worker(QThread):
    """Ejecuta `fn(*args, **kwargs)` en un hilo separado.

    Señales:
        progress(str): mensaje de estado opcional emitido por `fn` si acepta
            un callback `report_progress` como primer argumento posicional.
        finished_ok(object): resultado retornado por `fn`.
        failed(str): mensaje de error si `fn` lanzó una excepción.
    """

    progress = Signal(str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable, *args: Any, pass_progress: bool = False, **kwargs: Any):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._pass_progress = pass_progress

    def run(self):
        com_initialized = self._com_initialize()
        try:
            if self._pass_progress:
                result = self._fn(self.progress.emit, *self._args, **self._kwargs)
            else:
                result = self._fn(*self._args, **self._kwargs)
            self.finished_ok.emit(result)
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            if com_initialized:
                self._com_uninitialize()

    @staticmethod
    def _com_initialize() -> bool:
        """Inicializa el apartamento COM en este hilo (Windows/comtypes).

        COM requiere que cada hilo que lo use llame a CoInitialize antes de
        crear/usar proxies COM (como el SapObject de SAP2000); sin esto,
        las llamadas desde este QThread pueden fallar con errores de
        marshaling contra un SAP2000 real. En Linux/entornos sin comtypes
        (usados con FakeSapModel) esto no aplica y se omite en silencio.
        """
        try:
            import comtypes
            comtypes.CoInitialize()
            return True
        except Exception:
            return False

    @staticmethod
    def _com_uninitialize() -> None:
        try:
            import comtypes
            comtypes.CoUninitialize()
        except Exception:
            pass
