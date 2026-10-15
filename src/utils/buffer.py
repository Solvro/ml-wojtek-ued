class LevelBuffer:
    def __init__(self, size: int):
        self.size = size
        self._entries: list[tuple[float, object]] = []

    def add(self, value, score: float = float("-inf")) -> bool:
        if len(self._entries) < self.size:
            self._entries.append((score, value))
            return True
        weakest = min(range(len(self._entries)), key=lambda i: self._entries[i][0])
        if score <= self._entries[weakest][0]:
            return False
        self._entries[weakest] = (score, value)
        return True

    def get(self):
        return [value for _, value in self._entries]

    def clear(self) -> None:
        self._entries = []
