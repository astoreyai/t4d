"""
NCA Explorer - Rich terminal UI for browsing Neuro-Cognitive Architecture state.

Provides interactive exploration of:
- Neurotransmitter state (DA, 5-HT, ACh, NE, GABA, Glu)
- Energy landscape and attractor basins
- Coupling matrix dynamics
- Oscillation and sleep/wake state
- Forward-Forward and Capsule network state
- Glymphatic clearance metrics
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np

try:
    from rich import box
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# NCA imports
from t4dm.nca import (
    AdenosineConfig,
    AdenosineDynamics,
    AttractorBasin,
    CapsuleConfig,
    CapsuleNetwork,
    CognitiveState,
    CouplingConfig,
    EnergyConfig,
    EnergyLandscape,
    ForwardForwardConfig,
    ForwardForwardNetwork,
    FrequencyBandGenerator,
    GlymphaticConfig,
    GlymphaticSystem,
    LearnableCoupling,
    NeuralFieldConfig,
    NeuralFieldSolver,
    NeurotransmitterState,
    OscillatorConfig,
    StabilityAnalyzer,
    StabilityConfig,
    StateTransitionManager,
)

# NT labels and colors
NT_LABELS = ["DA", "5-HT", "ACh", "NE", "GABA", "Glu"]
NT_COLORS = ["red", "blue", "green", "yellow", "magenta", "cyan"]
NT_DESCRIPTIONS = [
    "Dopamine (reward/motivation)",
    "Serotonin (patience/mood)",
    "Acetylcholine (attention/learning)",
    "Norepinephrine (arousal/alertness)",
    "GABA (inhibition/calm)",
    "Glutamate (excitation/plasticity)",
]


@dataclass
class NCASnapshot:
    """Snapshot of NCA system state."""
    timestamp: datetime
    nt_state: np.ndarray  # [6]
    energy: float
    attractor: str | None
    stability_type: str | None
    oscillation_phase: str | None
    temperature: float
    beta: float


class NCAExplorer:
    """
    Interactive NCA state browser with Rich terminal UI.

    Features:
    - View NT concentrations with bar visualization
    - Explore energy landscape and attractors
    - Inspect coupling matrix
    - Monitor oscillation state
    - Track Forward-Forward learning
    - Observe Capsule network dynamics
    - Review Glymphatic clearance
    """

    def __init__(
        self,
        console: Optional["Console"] = None,
        auto_init: bool = True,
    ):
        """
        Initialize NCA explorer.

        Args:
            console: Rich console instance (creates new if None)
            auto_init: Automatically initialize NCA components
        """
        if not RICH_AVAILABLE:
            raise ImportError(
                "rich library required for NCAExplorer. "
                "Install with: pip install rich"
            )

        self.console = console or Console()

        # Core NCA components
        self._coupling: LearnableCoupling | None = None
        self._state_manager: StateTransitionManager | None = None
        self._energy: EnergyLandscape | None = None
        self._field: NeuralFieldSolver | None = None
        self._stability: StabilityAnalyzer | None = None
        self._oscillator: FrequencyBandGenerator | None = None
        self._adenosine: AdenosineDynamics | None = None

        # Phase 3-4 components
        self._ff_network: ForwardForwardNetwork | None = None
        self._capsule_network: CapsuleNetwork | None = None
        self._glymphatic: GlymphaticSystem | None = None

        # Current NT state
        self._nt_state = np.full(6, 0.5, dtype=np.float32)

        # History for tracking
        self._history: list[NCASnapshot] = []
        self._max_history = 1000

        self._initialized = False

        if auto_init:
            self._init_components()

    def _init_components(self) -> None:
        """Initialize NCA components with default configurations."""
        # Coupling matrix
        self._coupling = LearnableCoupling(CouplingConfig())

        # State manager with default attractors
        self._state_manager = StateTransitionManager()
        self._state_manager.register_attractor(
            CognitiveState.EXPLORATION,
            AttractorBasin(
                center=np.array([0.7, 0.3, 0.6, 0.7, 0.3, 0.6], dtype=np.float32),
                width=0.3,
                stability=1.0,
            ),
        )
        self._state_manager.register_attractor(
            CognitiveState.EXPLOITATION,
            AttractorBasin(
                center=np.array([0.5, 0.6, 0.4, 0.3, 0.5, 0.4], dtype=np.float32),
                width=0.3,
                stability=1.0,
            ),
        )
        self._state_manager.register_attractor(
            CognitiveState.REST,
            AttractorBasin(
                center=np.array([0.3, 0.5, 0.3, 0.2, 0.7, 0.3], dtype=np.float32),
                width=0.3,
                stability=1.0,
            ),
        )

        # Energy landscape
        self._energy = EnergyLandscape(
            config=EnergyConfig(),
            coupling=self._coupling,
            state_manager=self._state_manager,
        )

        # Neural field solver
        self._field = NeuralFieldSolver(NeuralFieldConfig())

        # Stability analyzer
        self._stability = StabilityAnalyzer(StabilityConfig())

        # Oscillator
        self._oscillator = FrequencyBandGenerator(OscillatorConfig())

        # Adenosine/sleep dynamics
        self._adenosine = AdenosineDynamics(AdenosineConfig())

        # Forward-Forward network
        self._ff_network = ForwardForwardNetwork(
            ForwardForwardConfig(layer_sizes=[64, 128, 64])
        )

        # Capsule network
        self._capsule_network = CapsuleNetwork(
            CapsuleConfig(
                input_dim=64,
                n_primary_capsules=8,
                n_digit_capsules=4,
            )
        )

        # Glymphatic system
        self._glymphatic = GlymphaticSystem(GlymphaticConfig())

        self._initialized = True

    def _record_snapshot(self) -> None:
        """Record current state to history."""
        if not self._initialized:
            return

        # Get current attractor
        attractor = None
        if self._state_manager:
            attractor = self._state_manager.get_current_state(self._nt_state)
            attractor = attractor.name if attractor else None

        # Get stability type
        stability_type = None
        if self._stability:
            result = self._stability.analyze(self._nt_state, self._coupling.K)
            stability_type = result.stability_type.name

        # Get oscillation phase
        osc_phase = None
        if self._oscillator:
            osc_phase = self._oscillator.state.phase.name if hasattr(self._oscillator, 'state') else None

        snapshot = NCASnapshot(
            timestamp=datetime.now(),
            nt_state=self._nt_state.copy(),
            energy=self._energy.compute_total_energy(self._nt_state) if self._energy else 0.0,
            attractor=attractor,
            stability_type=stability_type,
            oscillation_phase=osc_phase,
            temperature=self._energy._temperature if self._energy else 1.0,
            beta=self._energy.get_current_beta() if self._energy else 8.0,
        )

        self._history.append(snapshot)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    # =========================================================================
    # NT State Visualization
    # =========================================================================

    def _render_nt_bars(self) -> Table:
        """Render NT concentrations as colored bars."""
        table = Table(title="Neurotransmitter State", box=box.ROUNDED, show_header=True)
        table.add_column("NT", style="bold", width=6)
        table.add_column("Level", justify="right", width=6)
        table.add_column("Bar", width=30)
        table.add_column("Description", style="dim", width=30)

        for i, (label, color, desc) in enumerate(zip(NT_LABELS, NT_COLORS, NT_DESCRIPTIONS)):
            level = self._nt_state[i]
            bar_width = int(level * 20)
            bar = f"[{color}]" + "█" * bar_width + "░" * (20 - bar_width) + f"[/{color}]"

            table.add_row(
                f"[{color}]{label}[/{color}]",
                f"{level:.2f}",
                bar,
                desc,
            )

        return table

    def _render_nt_radar(self) -> Panel:
        """Render NT state as ASCII radar plot."""
        # Simple ASCII radar representation
        lines = []
        lines.append("      NT Radar")
        lines.append("    ┌─────────┐")

        # Top row: NE
        ne_level = int(self._nt_state[3] * 5)
        lines.append(f"    │  NE:{ne_level}   │")

        # Middle: DA left, ACh right
        da_level = int(self._nt_state[0] * 5)
        ach_level = int(self._nt_state[2] * 5)
        lines.append(f"  DA:{da_level}│    ●    │ACh:{ach_level}")

        # Bottom middle: 5-HT left, Glu right
        sht_level = int(self._nt_state[1] * 5)
        glu_level = int(self._nt_state[5] * 5)
        lines.append(f" 5HT:{sht_level}│         │Glu:{glu_level}")

        # Bottom: GABA
        gaba_level = int(self._nt_state[4] * 5)
        lines.append(f"    │ GABA:{gaba_level} │")
        lines.append("    └─────────┘")

        text = Text("\n".join(lines))
        return Panel(text, title="Radar View", border_style="cyan")

    def show_nt_state(self) -> None:
        """Display current NT state."""
        layout = Layout()
        layout.split_row(
            Layout(name="bars", ratio=2),
            Layout(name="radar", ratio=1),
        )

        layout["bars"].update(Panel(self._render_nt_bars(), border_style="green"))
        layout["radar"].update(self._render_nt_radar())

        self.console.print(layout)

    # =========================================================================
    # Energy Landscape
    # =========================================================================

    def _render_energy_summary(self) -> Table:
        """Render energy landscape summary."""
        if not self._energy:
            return Table(title="Energy Landscape (not initialized)")

        table = Table(title="Energy Landscape", box=box.ROUNDED)
        table.add_column("Component", style="bold magenta")
        table.add_column("Value", justify="right", style="white")

        # Compute energies
        e_hop = self._energy.compute_hopfield_energy(self._nt_state)
        e_bound = self._energy.compute_boundary_energy(self._nt_state)
        e_attr = self._energy.compute_attractor_energy(self._nt_state)
        e_total = e_hop + e_bound + e_attr

        table.add_row("Hopfield Energy", f"{e_hop:.4f}")
        table.add_row("Boundary Energy", f"{e_bound:.4f}")
        table.add_row("Attractor Energy", f"{e_attr:.4f}")
        table.add_section()
        table.add_row("[bold]Total Energy[/bold]", f"[bold]{e_total:.4f}[/bold]")
        table.add_section()
        table.add_row("Temperature", f"{self._energy._temperature:.3f}")
        table.add_row("Beta (Hopfield)", f"{self._energy.get_current_beta():.2f}")

        # Gradient norm
        grad = self._energy.compute_energy_gradient(self._nt_state)
        grad_norm = float(np.linalg.norm(grad))
        table.add_row("Gradient Norm", f"{grad_norm:.4f}")

        return table

    def _render_attractor_table(self) -> Table:
        """Render attractor basin information."""
        if not self._state_manager:
            return Table(title="Attractors (not initialized)")

        table = Table(title="Attractor Basins", box=box.ROUNDED)
        table.add_column("State", style="bold cyan")
        table.add_column("Distance", justify="right")
        table.add_column("Width")
        table.add_column("Stability")
        table.add_column("Active", justify="center")

        current_state = self._state_manager.get_current_state(self._nt_state)

        for state, attractor in self._state_manager.attractors.items():
            dist = float(np.linalg.norm(self._nt_state - attractor.center))
            is_active = state == current_state

            active_marker = "[green]●[/green]" if is_active else "[dim]○[/dim]"

            table.add_row(
                state.name,
                f"{dist:.3f}",
                f"{attractor.width:.2f}",
                f"{attractor.stability:.2f}",
                active_marker,
            )

        return table

    def show_energy_landscape(self) -> None:
        """Display energy landscape state."""
        layout = Layout()
        layout.split_row(
            Layout(name="energy", ratio=1),
            Layout(name="attractors", ratio=1),
        )

        layout["energy"].update(Panel(self._render_energy_summary(), border_style="yellow"))
        layout["attractors"].update(Panel(self._render_attractor_table(), border_style="blue"))

        self.console.print(layout)

    # =========================================================================
    # Coupling Matrix
    # =========================================================================

    def _render_coupling_matrix(self) -> Table:
        """Render coupling matrix as table."""
        if not self._coupling:
            return Table(title="Coupling Matrix (not initialized)")

        K = self._coupling.K
        table = Table(title="Coupling Matrix K", box=box.ROUNDED)

        # Header
        table.add_column("", style="bold")
        for label in NT_LABELS:
            table.add_column(label, justify="right", width=7)

        # Rows
        for i, label in enumerate(NT_LABELS):
            row = [f"[bold]{label}[/bold]"]
            for j in range(6):
                val = K[i, j]
                # Color based on value
                if val > 0.3:
                    color = "green"
                elif val < -0.3:
                    color = "red"
                else:
                    color = "white"
                row.append(f"[{color}]{val:+.2f}[/{color}]")
            table.add_row(*row)

        return table

    def _render_coupling_stats(self) -> Table:
        """Render coupling matrix statistics."""
        if not self._coupling:
            return Table(title="Coupling Stats (not initialized)")

        K = self._coupling.K
        table = Table(title="Coupling Statistics", box=box.ROUNDED, show_header=False)
        table.add_column("Metric", style="bold magenta")
        table.add_column("Value", style="white")

        # Spectral analysis
        eigenvalues = np.linalg.eigvals(K)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        max_real = float(np.max(eigenvalues.real))

        table.add_row("Spectral Radius", f"{spectral_radius:.4f}")
        table.add_row("Max Real Eigenvalue", f"{max_real:.4f}")
        table.add_row("Trace", f"{float(np.trace(K)):.4f}")
        table.add_row("Frobenius Norm", f"{float(np.linalg.norm(K, 'fro')):.4f}")

        # Symmetry check
        asymmetry = float(np.linalg.norm(K - K.T))
        table.add_row("Asymmetry", f"{asymmetry:.4f}")

        # E/I balance
        excitatory = float(np.sum(K[K > 0]))
        inhibitory = float(np.sum(K[K < 0]))
        table.add_row("Total Excitation", f"{excitatory:.3f}")
        table.add_row("Total Inhibition", f"{inhibitory:.3f}")
        table.add_row("E/I Ratio", f"{excitatory / abs(inhibitory) if inhibitory != 0 else np.inf:.3f}")

        return table

    def show_coupling_matrix(self) -> None:
        """Display coupling matrix."""
        layout = Layout()
        layout.split_row(
            Layout(name="matrix", ratio=2),
            Layout(name="stats", ratio=1),
        )

        layout["matrix"].update(Panel(self._render_coupling_matrix(), border_style="green"))
        layout["stats"].update(Panel(self._render_coupling_stats(), border_style="magenta"))

        self.console.print(layout)

    # =========================================================================
    # Stability Analysis
    # =========================================================================

    def show_stability(self) -> None:
        """Display stability analysis."""
        if not self._stability or not self._coupling:
            self.console.print("[red]Stability analyzer not initialized[/red]")
            return

        result = self._stability.analyze(self._nt_state, self._coupling.K)

        table = Table(title="Stability Analysis", box=box.ROUNDED)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="white")

        # Stability type with color
        stability_colors = {
            "STABLE": "green",
            "MARGINALLY_STABLE": "yellow",
            "UNSTABLE": "red",
            "OSCILLATORY": "blue",
        }
        st_color = stability_colors.get(result.stability_type.name, "white")
        table.add_row("Stability Type", f"[{st_color}]{result.stability_type.name}[/{st_color}]")

        table.add_row("Lyapunov Exponent", f"{result.lyapunov_exponent:.4f}")
        table.add_row("Spectral Radius", f"{result.spectral_radius:.4f}")
        table.add_row("Max Real Eigenvalue", f"{result.max_real_eigenvalue:.4f}")

        if result.dominant_frequency:
            table.add_row("Dominant Frequency", f"{result.dominant_frequency:.2f} Hz")

        self.console.print(Panel(table, border_style="cyan"))

    # =========================================================================
    # Oscillation State
    # =========================================================================

    def show_oscillations(self) -> None:
        """Display oscillation state."""
        if not self._oscillator:
            self.console.print("[red]Oscillator not initialized[/red]")
            return

        table = Table(title="Oscillation State", box=box.ROUNDED)
        table.add_column("Band", style="bold")
        table.add_column("Frequency", justify="right")
        table.add_column("Power", justify="right")
        table.add_column("Phase", justify="right")

        # Generate current oscillation state
        bands = [
            ("Delta", "0.5-4 Hz", "cyan"),
            ("Theta", "4-8 Hz", "green"),
            ("Alpha", "8-12 Hz", "yellow"),
            ("Beta", "12-30 Hz", "magenta"),
            ("Gamma", "30-100 Hz", "red"),
        ]

        for name, freq_range, color in bands:
            # Simulated values for display
            power = np.random.uniform(0.1, 1.0)
            phase = np.random.uniform(0, 2 * np.pi)
            table.add_row(
                f"[{color}]{name}[/{color}]",
                freq_range,
                f"{power:.3f}",
                f"{phase:.2f} rad",
            )

        self.console.print(Panel(table, border_style="blue"))

        # Sleep state
        if self._adenosine:
            sleep_table = Table(title="Sleep/Wake State", box=box.ROUNDED, show_header=False)
            sleep_table.add_column("Metric", style="bold magenta")
            sleep_table.add_column("Value", style="white")

            state = self._adenosine.state
            sleep_table.add_row("Adenosine Level", f"{state.adenosine_level:.3f}")
            sleep_table.add_row("Sleep Pressure", f"{state.sleep_pressure:.3f}")
            sleep_table.add_row("Wake Duration", f"{state.wake_duration:.1f} h")
            sleep_table.add_row("Sleep State", state.sleep_state.name)

            self.console.print(Panel(sleep_table, border_style="magenta"))

    # =========================================================================
    # Forward-Forward State
    # =========================================================================

    def show_forward_forward(self) -> None:
        """Display Forward-Forward network state."""
        if not self._ff_network:
            self.console.print("[red]Forward-Forward network not initialized[/red]")
            return

        table = Table(title="Forward-Forward Network", box=box.ROUNDED)
        table.add_column("Layer", style="bold cyan", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Goodness+", justify="right")
        table.add_column("Goodness-", justify="right")
        table.add_column("Threshold", justify="right")

        state = self._ff_network.state
        for i, layer in enumerate(self._ff_network.layers):
            layer_state = layer.state
            table.add_row(
                f"Layer {i}",
                str(layer.config.hidden_size),
                f"{layer_state.positive_goodness:.3f}",
                f"{layer_state.negative_goodness:.3f}",
                f"{layer.config.threshold:.2f}",
            )

        self.console.print(Panel(table, border_style="green"))

        # Stats
        stats_table = Table(title="FF Statistics", box=box.ROUNDED, show_header=False)
        stats_table.add_column("Metric", style="bold")
        stats_table.add_column("Value")

        stats_table.add_row("Training Steps", str(state.training_step))
        stats_table.add_row("Mean Margin", f"{np.mean(state.margins):.4f}" if state.margins else "N/A")
        stats_table.add_row("Phase", state.phase.name)

        self.console.print(Panel(stats_table, border_style="yellow"))

    # =========================================================================
    # Capsule Network State
    # =========================================================================

    def show_capsules(self) -> None:
        """Display Capsule network state."""
        if not self._capsule_network:
            self.console.print("[red]Capsule network not initialized[/red]")
            return

        table = Table(title="Capsule Network", box=box.ROUNDED)
        table.add_column("Capsule", style="bold")
        table.add_column("Probability", justify="right")
        table.add_column("Pose Norm", justify="right")
        table.add_column("Routing", justify="right")

        state = self._capsule_network.state
        for i in range(min(8, len(state.entity_probs))):
            prob = state.entity_probs[i]
            pose_norm = np.linalg.norm(state.poses[i]) if len(state.poses) > i else 0.0
            routing = np.mean(state.routing_weights[i]) if len(state.routing_weights) > i else 0.0

            # Color by probability
            color = "green" if prob > 0.7 else "yellow" if prob > 0.3 else "red"

            table.add_row(
                f"Capsule {i}",
                f"[{color}]{prob:.3f}[/{color}]",
                f"{pose_norm:.3f}",
                f"{routing:.3f}",
            )

        self.console.print(Panel(table, border_style="magenta"))

        # Routing entropy
        if state.routing_weights.size > 0:
            entropy = -np.sum(state.routing_weights * np.log(state.routing_weights + 1e-8))
            self.console.print(f"Routing Entropy: {entropy:.4f}")

    # =========================================================================
    # Glymphatic State
    # =========================================================================

    def show_glymphatic(self) -> None:
        """Display Glymphatic system state."""
        if not self._glymphatic:
            self.console.print("[red]Glymphatic system not initialized[/red]")
            return

        table = Table(title="Glymphatic System", box=box.ROUNDED)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="white")

        state = self._glymphatic.state

        table.add_row("Sleep Stage", state.sleep_stage.name)
        table.add_row("Clearance Rate", f"{state.clearance_rate:.4f}")
        table.add_row("CSF Flow", f"{state.csf_flow:.4f}")
        table.add_row("Interstitial Space", f"{state.interstitial_space:.4f}")

        # Waste levels
        table.add_section()
        table.add_row("[bold]Waste Levels[/bold]", "")
        for waste_type, level in state.waste_levels.items():
            color = "green" if level < 0.3 else "yellow" if level < 0.7 else "red"
            table.add_row(f"  {waste_type}", f"[{color}]{level:.3f}[/{color}]")

        self.console.print(Panel(table, border_style="blue"))

    # =========================================================================
    # Interactive Mode
    # =========================================================================

    def set_nt_state(self, nt_idx: int, value: float) -> None:
        """Set a specific NT value."""
        if 0 <= nt_idx < 6:
            self._nt_state[nt_idx] = np.clip(value, 0.0, 1.0)
            self._record_snapshot()

    def step_dynamics(self, dt: float = 0.01) -> None:
        """Step NT dynamics forward."""
        if self._energy:
            self._nt_state = self._energy.langevin_step(self._nt_state, dt=dt)
            self._record_snapshot()

    def relax_to_equilibrium(self, max_steps: int = 100) -> tuple[int, float]:
        """Relax system to energy minimum."""
        if not self._energy:
            return 0, 0.0

        self._nt_state, steps, energy = self._energy.relax_to_equilibrium(
            self._nt_state, max_steps=max_steps
        )
        self._record_snapshot()
        return steps, energy

    def interactive(self) -> None:
        """
        Run interactive NCA explorer session.

        Provides menu-driven interface for exploring NCA state.
        """
        self.console.print(Panel(
            "[bold cyan]T4DM NCA Explorer[/bold cyan]\n"
            "Interactive neural dynamics exploration interface",
            border_style="cyan",
        ))

        while True:
            self.console.print("\n[bold magenta]Options:[/bold magenta]")
            self.console.print("1. View NT State")
            self.console.print("2. View Energy Landscape")
            self.console.print("3. View Coupling Matrix")
            self.console.print("4. View Stability Analysis")
            self.console.print("5. View Oscillations")
            self.console.print("6. View Forward-Forward")
            self.console.print("7. View Capsules")
            self.console.print("8. View Glymphatic")
            self.console.print("9. Set NT Value")
            self.console.print("10. Step Dynamics")
            self.console.print("11. Relax to Equilibrium")
            self.console.print("12. Full Dashboard")
            self.console.print("0. Exit")

            choice = Prompt.ask(
                "\n[bold cyan]Select option[/bold cyan]",
                choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
                default="0",
            )

            if choice == "1":
                self.show_nt_state()
            elif choice == "2":
                self.show_energy_landscape()
            elif choice == "3":
                self.show_coupling_matrix()
            elif choice == "4":
                self.show_stability()
            elif choice == "5":
                self.show_oscillations()
            elif choice == "6":
                self.show_forward_forward()
            elif choice == "7":
                self.show_capsules()
            elif choice == "8":
                self.show_glymphatic()
            elif choice == "9":
                nt_idx = int(Prompt.ask(
                    "NT index (0=DA, 1=5-HT, 2=ACh, 3=NE, 4=GABA, 5=Glu)",
                    choices=["0", "1", "2", "3", "4", "5"],
                ))
                value = float(Prompt.ask("Value (0.0-1.0)", default="0.5"))
                self.set_nt_state(nt_idx, value)
                self.console.print(f"[green]Set {NT_LABELS[nt_idx]} = {value:.2f}[/green]")
            elif choice == "10":
                steps = int(Prompt.ask("Number of steps", default="10"))
                for _ in range(steps):
                    self.step_dynamics()
                self.console.print(f"[green]Stepped {steps} dynamics iterations[/green]")
                self.show_nt_state()
            elif choice == "11":
                steps, energy = self.relax_to_equilibrium()
                self.console.print(f"[green]Relaxed in {steps} steps to energy {energy:.4f}[/green]")
                self.show_nt_state()
            elif choice == "12":
                self.show_dashboard()
            elif choice == "0":
                self.console.print("\n[bold green]Goodbye![/bold green]")
                break

    def show_dashboard(self) -> None:
        """Display full NCA dashboard."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )

        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        layout["left"].split_column(
            Layout(name="nt", ratio=1),
            Layout(name="energy", ratio=1),
        )

        layout["right"].split_column(
            Layout(name="coupling", ratio=1),
            Layout(name="stats", ratio=1),
        )

        # Header
        header_text = Text("T4DM NCA Dashboard", style="bold cyan")
        layout["header"].update(Panel(header_text, border_style="cyan"))

        # Panels
        layout["nt"].update(Panel(self._render_nt_bars(), border_style="green"))
        layout["energy"].update(Panel(self._render_energy_summary(), border_style="yellow"))
        layout["coupling"].update(Panel(self._render_coupling_matrix(), border_style="magenta"))
        layout["stats"].update(Panel(self._render_coupling_stats(), border_style="blue"))

        # Footer
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        footer_text = Text(f"Snapshot: {now} | History: {len(self._history)} records", style="dim")
        layout["footer"].update(Panel(footer_text, border_style="dim"))

        self.console.print(layout)

    def get_stats(self) -> dict:
        """Get NCA explorer statistics."""
        return {
            "initialized": self._initialized,
            "history_size": len(self._history),
            "current_nt_state": self._nt_state.tolist(),
            "current_energy": self._energy.compute_total_energy(self._nt_state) if self._energy else None,
            "current_attractor": (
                self._state_manager.get_current_state(self._nt_state).name
                if self._state_manager else None
            ),
        }


def main():
    """CLI entry point for NCA explorer."""
    explorer = NCAExplorer()
    explorer.interactive()


if __name__ == "__main__":
    main()
