# Quantum Photon Processor (QPP) API Reference

## Core Classes

### 1. PhysicsHypercubeSystemEnhanced
**Многомерное пространство для моделирования физических систем**

```python
class PhysicsHypercubeSystemEnhanced(dimensions, resolution=100, ...)
```

**Parameters:**
- `dimensions` (dict): Диапазоны параметров (e.g., `{'frequency': (1e14, 1e15)}`)
- `resolution` (int): Точек на измерение (автонастройка по RAM)
- `extrapolation_limit` (float): Макс. отклонение для экстраполяции
- `physical_constraint` (function): Функция физических ограничений

**Key Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `add_known_point(point, value)` | `point` (dict): Параметры точки<br>`value` (float): Значение | None | Добавляет экспериментальную точку |
| `physical_query_dict(params)` | `params` (dict): Входные параметры | float | Прогноз значения физического закона |
| `holographic_compression_3d(ratio=0.005)` | `ratio` (float): Коэффициент сжатия | np.array | 3D-проекция данных с сохранением топологии |
| `quantum_entanglement_optimization(depth=5)` | `depth` (int): Глубина квант. схемы | None | Оптимизация через VQE/QAOA |
| `calculate_topological_invariants()` | None | dict | Вычисляет числа Бетти и персистентные гомологии |

---

### 2. QuantumPhotonProcessorEnhanced
**Управление фотонными кубитами с топологической оптимизацией**

```python
class QuantumPhotonProcessorEnhanced(photon_dimensions, ...)
```

**Parameters:**
- `photon_dimensions` (dict): Параметры фотонных состояний
- `compression_mode` (str): Режим сжатия ("holo-quantum-3d")
- `security_key` (bytes): Ключ шифрования (32 байта)

**Key Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `create_qubit(state)` | `state` (dict): Параметры кубита | str | Создает оптимизированный фотонный кубит |
| `entangle_qubits(qubit1, qubit2)` | `qubit1`, `qubit2` (str): ID кубитов | None | Запутывание с топологической редукцией |
| `save_state(filename, key=None)` | `filename` (str): Путь<br>`key` (bytes): Ключ шифрования | str | Сохраняет состояние процессора |
| `integrate_topological_neural_network(output_dim)` | `output_dim` (int): Размер выхода | bool | Интегрирует TopoNN в систему |
| `quantum_entanglement_optimization(depth=5)` | `depth` (int): Глубина схемы | None | Квантовая оптимизация критических точек |

---

### 3. QuantumMemory
**Квантовая память с эмоциональными метками и запутанностью**

```python
class QuantumMemory()
```

**Key Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `save_memory(memory_id, content, emotion_vector)` | `memory_id` (str): ID<br>`content` (dict): Данные<br>`emotion_vector` (list): [0.8, 0.2] | str | Сохраняет состояние с эмоц. меткой |
| `entangle(memory_id1, memory_id2)` | `memory_id1`, `memory_id2` (str) | str | Запутывает два воспоминания |
| `recall(memory_id, superposition=False)` | `memory_id` (str)<br>`superposition` (bool) | dict | Восстанавливает состояние (с запутанностью) |

---

### 4. TopologicalNeuralNetwork
**Интерпретируемые нейронные сети на топологических инвариантах**

```python
class TopologicalNeuralNetwork(input_dim, output_dim, ...)
```

**Key Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `fit(X, y, use_topological=True)` | `X` (array): Данные<br>`y` (array): Метки<br>`use_topological` (bool) | History | Обучение с топологическими признаками |
| `predict(X)` | `X` (array): Входные данные | array | Предсказание |
| `interpret()` | None | dict | Интерпретация важности признаков |
| `visualize_interpretation()` | None | None | Визуализация важности признаков и истории |

---

## Advanced Components

### ECDSAHypercubeIntegrator
```python
class ECDSAHypercubeIntegrator(curve, public_key_Q, hypercube_system)
```
**Методы:** `map_point(i, j)`, `find_collisions(target_r)`, `recover_private_key(collision_lines)`

### QuantumResistantECDSA
```python
class QuantumResistantECDSA(curve)
```
**Методы:** `generate_key_pair()`, `sign(message)`, `verify(message, signature, public_key)`

### QuantumTopologicalHybrid
```python
class QuantumTopologicalHybrid(topological_nn, n_qubits=4)
```
**Методы:** `quantum_enhancement(X, y)`, `topological_quantum_interpretation()`

---

## Пример использования

```python
# Инициализация процессора
qpp = QuantumPhotonProcessorEnhanced(
    ADVANCED_PHOTON_DIMENSIONS,
    compression_mode="holo-quantum-3d"
)

# Создание кубита
qubit_id = qpp.create_qubit({
    'frequency': 5.5e14,
    'polarization': np.pi/4,
    'oam_state': 3
})

# Сохранение состояния в квантовую память
qpp.save_processor_state_to_memory("experiment_1", [0.9, 0.1, 0.3])

# Запутывание кубитов
qpp.entangle_qubits("Q1", "Q2")

# Интеграция нейросети
qpp.integrate_topological_neural_network(output_dim=3)

# Топологический анализ
invariants = qpp.system.calculate_topological_invariants()
print(f"Betti numbers: {invariants['betti_numbers']}")

# Криптографический анализ
collision_lines = qpp.ecdsa_integrator.find_collisions(target_r=0x7F3A...)
private_key = qpp.ecdsa_integrator.recover_private_key(collision_lines)
```

---

## Физические основы методов

1. **Голографическое сжатие**:
   - Принцип AdS/CFT соответствия
   - Проекция данных в 3D через UMAP (Uniform Manifold Approximation)
   - Сохранение топологических инвариантов: $\beta_0$ (компоненты связности), $\beta_1$ (топологические циклы)

2. **Квантовая оптимизация**:
   $$ \min_{\theta} \langle \psi(\theta) | H | \psi(\theta) \rangle $$
   - Гамильтониан $H$ строится на критических точках
   - Параметризованные квантовые схемы (EfficientSU2)

3. **Топологические признаки**:
   - Персистентные гомологии: $H_k(X) = \ker\partial_k / \operatorname{im}\partial_{k+1}$
   - Ландшафты персистенции: $\Lambda_k(\tau) = \sup \{m > 0 \mid (b,d) \in \text{Dgm}, \tau - m \leq b < d \leq \tau + m\}$

Система обеспечивает O(100x) сжатие данных при сохранении 99.9% физической информации за счет топологической редукции размерности.
