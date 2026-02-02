"""
Learning Inspector - Rich terminal UI for inspecting T4DM learning dynamics.

Provides interactive exploration of:
- Dopamine reward prediction errors (RPE)
- Serotonin long-term credit assignment
- Norepinephrine arousal and attention
- Acetylcholine encoding/retrieval mode
- Eligibility traces and credit flow
- Three-factor learning signals
- FSRS spaced repetition scheduling
- Causal discovery and attribution
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID

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
    from rich.tree import Tree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Learning imports
from t4dm.learning import (
    # FSRS
    FSRS,
    AcetylcholineState,
    # Acetylcholine
    AcetylcholineSystem,
    ArousalState,
    CausalAttributor,
    CausalGraph,
    # Causal
    CausalLearner,
    CognitiveMode,
    # Credit flow
    CreditFlowEngine,
    # Dopamine
    DopamineSystem,
    EligibilityConfig,
    # Eligibility
    EligibilityTrace,
    # Events
    Experience,
    FSRSMemoryTracker,
    LayeredEligibilityTrace,
    MemoryState,
    # Neuromodulators
    NeuromodulatorOrchestra,
    NeuromodulatorState,
    # Norepinephrine
    NorepinephrineSystem,
    OutcomeEvent,
    OutcomeType,
    Rating,
    RetrievalEvent,
    RewardPredictionError,
    # Serotonin
    SerotoninSystem,
    TemporalContext,
    # Three-factor
    ThreeFactorLearningRule,
    ThreeFactorSignal,
    compute_rpe,
    create_fsrs,
    create_neuromodulator_orchestra,
    create_three_factor_rule,
)

# Neuromodulator colors and descriptions
NM_LABELS = ["DA", "5-HT", "NE", "ACh"]
NM_COLORS = ["red", "blue", "yellow", "green"]
NM_DESCRIPTIONS = [
    "Dopamine: Reward prediction error",
    "Serotonin: Long-term credit, patience",
    "Norepinephrine: Arousal, attention",
    "Acetylcholine: Encoding/retrieval mode",
]


@dataclass
class LearningSnapshot:
    """Snapshot of learning system state."""
    timestamp: datetime
    dopamine_rpe: float
    serotonin_credit: float
    norepinephrine_arousal: float
    acetylcholine_mode: str
    learning_rate_multiplier: float
    eligibility_mean: float


class LearningInspector:
    """
    Interactive learning dynamics inspector with Rich terminal UI.

    Features:
    - View neuromodulator state (DA, 5-HT, NE, ACh)
    - Inspect eligibility traces
    - Monitor credit flow and three-factor learning
    - Track FSRS scheduling for memories
    - Explore causal attribution
    - Simulate learning events
    """

    def __init__(
        self,
        session_id: str | None = None,
        console: Optional["Console"] = None,
        auto_init: bool = True,
    ):
        """
        Initialize learning inspector.

        Args:
            session_id: Session to inspect
            console: Rich console instance (creates new if None)
            auto_init: Automatically initialize learning components
        """
        if not RICH_AVAILABLE:
            raise ImportError(
                "rich library required for LearningInspector. "
                "Install with: pip install rich"
            )

        self.session_id = session_id
        self.console = console or Console()

        # Learning components
        self._orchestra: NeuromodulatorOrchestra | None = None
        self._dopamine: DopamineSystem | None = None
        self._serotonin: SerotoninSystem | None = None
        self._norepinephrine: NorepinephrineSystem | None = None
        self._acetylcholine: AcetylcholineSystem | None = None

        # Learning mechanisms
        self._eligibility: LayeredEligibilityTrace | None = None
        self._three_factor: ThreeFactorLearningRule | None = None
        self._credit_flow: CreditFlowEngine | None = None

        # FSRS tracking
        self._fsrs: FSRS | None = None
        self._memory_tracker: FSRSMemoryTracker | None = None

        # Causal discovery
        self._causal_learner: CausalLearner | None = None
        self._causal_graph: CausalGraph | None = None

        # History tracking
        self._history: list[LearningSnapshot] = []
        self._max_history = 1000
        self._events: list[Experience] = []

        self._initialized = False

        if auto_init:
            self._init_components()

    def _init_components(self) -> None:
        """Initialize learning components with default configurations."""
        # Neuromodulator orchestra
        self._orchestra = create_neuromodulator_orchestra()
        self._dopamine = self._orchestra.dopamine
        self._serotonin = self._orchestra.serotonin
        self._norepinephrine = self._orchestra.norepinephrine
        self._acetylcholine = self._orchestra.acetylcholine

        # Eligibility traces
        self._eligibility = LayeredEligibilityTrace(
            config=EligibilityConfig(tau=0.5, decay_type="exponential"),
            n_layers=3,
        )

        # Three-factor learning
        self._three_factor = create_three_factor_rule()

        # FSRS
        self._fsrs = create_fsrs()
        self._memory_tracker = FSRSMemoryTracker(fsrs=self._fsrs)

        # Causal discovery
        self._causal_learner = CausalLearner()
        self._causal_graph = CausalGraph()

        self._initialized = True

    def _record_snapshot(self) -> None:
        """Record current learning state to history."""
        if not self._initialized:
            return

        snapshot = LearningSnapshot(
            timestamp=datetime.now(),
            dopamine_rpe=self._dopamine.rpe if self._dopamine else 0.0,
            serotonin_credit=self._serotonin.credit if self._serotonin else 0.0,
            norepinephrine_arousal=self._norepinephrine.arousal if self._norepinephrine else 0.5,
            acetylcholine_mode=self._acetylcholine.mode.name if self._acetylcholine else "RETRIEVAL",
            learning_rate_multiplier=self._three_factor.compute_multiplier() if self._three_factor else 1.0,
            eligibility_mean=self._eligibility.mean() if self._eligibility else 0.0,
        )

        self._history.append(snapshot)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    # =========================================================================
    # Neuromodulator Visualization
    # =========================================================================

    def _render_neuromodulator_state(self) -> Table:
        """Render neuromodulator state as table with bars."""
        table = Table(title="Neuromodulator State", box=box.ROUNDED)
        table.add_column("System", style="bold", width=6)
        table.add_column("Level", justify="right", width=8)
        table.add_column("Bar", width=20)
        table.add_column("State", width=20)
        table.add_column("Description", style="dim", width=30)

        # Dopamine
        if self._dopamine:
            da_level = np.clip(self._dopamine.rpe + 0.5, 0, 1)  # Center at 0.5
            da_bar = self._level_bar(da_level, "red")
            da_state = f"RPE: {self._dopamine.rpe:+.3f}"
            table.add_row("[red]DA[/red]", f"{da_level:.2f}", da_bar, da_state, NM_DESCRIPTIONS[0])
        else:
            table.add_row("[red]DA[/red]", "N/A", "", "", NM_DESCRIPTIONS[0])

        # Serotonin
        if self._serotonin:
            sht_level = np.clip(self._serotonin.credit, 0, 1)
            sht_bar = self._level_bar(sht_level, "blue")
            sht_state = f"Credit: {self._serotonin.credit:.3f}"
            table.add_row("[blue]5-HT[/blue]", f"{sht_level:.2f}", sht_bar, sht_state, NM_DESCRIPTIONS[1])
        else:
            table.add_row("[blue]5-HT[/blue]", "N/A", "", "", NM_DESCRIPTIONS[1])

        # Norepinephrine
        if self._norepinephrine:
            ne_level = self._norepinephrine.arousal
            ne_bar = self._level_bar(ne_level, "yellow")
            ne_state = self._norepinephrine.state.name
            table.add_row("[yellow]NE[/yellow]", f"{ne_level:.2f}", ne_bar, ne_state, NM_DESCRIPTIONS[2])
        else:
            table.add_row("[yellow]NE[/yellow]", "N/A", "", "", NM_DESCRIPTIONS[2])

        # Acetylcholine
        if self._acetylcholine:
            ach_level = self._acetylcholine.level
            ach_bar = self._level_bar(ach_level, "green")
            ach_state = self._acetylcholine.mode.name
            table.add_row("[green]ACh[/green]", f"{ach_level:.2f}", ach_bar, ach_state, NM_DESCRIPTIONS[3])
        else:
            table.add_row("[green]ACh[/green]", "N/A", "", "", NM_DESCRIPTIONS[3])

        return table

    def _level_bar(self, level: float, color: str) -> str:
        """Create a colored level bar."""
        bar_width = int(level * 15)
        return f"[{color}]" + "█" * bar_width + "░" * (15 - bar_width) + f"[/{color}]"

    def show_neuromodulators(self) -> None:
        """Display neuromodulator state."""
        self.console.print(Panel(
            self._render_neuromodulator_state(),
            border_style="magenta",
        ))

        # Show detailed dopamine info
        if self._dopamine:
            da_table = Table(title="Dopamine Details", box=box.ROUNDED, show_header=False)
            da_table.add_column("Metric", style="bold red")
            da_table.add_column("Value", style="white")

            da_table.add_row("Reward Prediction Error", f"{self._dopamine.rpe:+.4f}")
            da_table.add_row("Expected Reward", f"{self._dopamine.expected_reward:.4f}")
            da_table.add_row("Actual Reward", f"{self._dopamine.actual_reward:.4f}")
            da_table.add_row("Learning Rate Mod", f"{self._dopamine.learning_rate_modifier:.4f}")

            self.console.print(Panel(da_table, border_style="red"))

    # =========================================================================
    # Eligibility Traces
    # =========================================================================

    def _render_eligibility_table(self) -> Table:
        """Render eligibility trace state."""
        table = Table(title="Eligibility Traces", box=box.ROUNDED)
        table.add_column("Layer", style="bold cyan")
        table.add_column("Active", justify="right")
        table.add_column("Mean", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Decay", justify="right")

        if self._eligibility:
            for i, layer in enumerate(self._eligibility.layers):
                n_active = int(np.sum(layer.traces > 0.01))
                mean_val = float(np.mean(layer.traces)) if layer.traces.size > 0 else 0.0
                max_val = float(np.max(layer.traces)) if layer.traces.size > 0 else 0.0

                table.add_row(
                    f"Layer {i}",
                    str(n_active),
                    f"{mean_val:.4f}",
                    f"{max_val:.4f}",
                    f"{layer.config.tau:.2f}",
                )
        else:
            table.add_row("N/A", "0", "0.0", "0.0", "0.0")

        return table

    def _render_eligibility_heatmap(self) -> Panel:
        """Render ASCII heatmap of eligibility traces."""
        if not self._eligibility or not self._eligibility.layers:
            return Panel(Text("No eligibility traces"), title="Trace Heatmap")

        # Get first layer traces
        traces = self._eligibility.layers[0].traces
        if traces.size == 0:
            return Panel(Text("Empty traces"), title="Trace Heatmap")

        # Create ASCII heatmap (simplified 10x10)
        n = min(10, len(traces))
        lines = []
        chars = " ░▒▓█"

        for i in range(n):
            row = ""
            for j in range(n):
                idx = i * n + j
                if idx < len(traces):
                    val = traces[idx]
                    char_idx = int(val * (len(chars) - 1))
                    row += chars[char_idx]
                else:
                    row += " "
            lines.append(row)

        text = Text("\n".join(lines))
        return Panel(text, title="Trace Heatmap (Layer 0)", border_style="cyan")

    def show_eligibility(self) -> None:
        """Display eligibility trace state."""
        layout = Layout()
        layout.split_row(
            Layout(name="table", ratio=2),
            Layout(name="heatmap", ratio=1),
        )

        layout["table"].update(Panel(self._render_eligibility_table(), border_style="cyan"))
        layout["heatmap"].update(self._render_eligibility_heatmap())

        self.console.print(layout)

    # =========================================================================
    # Three-Factor Learning
    # =========================================================================

    def _render_three_factor_table(self) -> Table:
        """Render three-factor learning state."""
        table = Table(title="Three-Factor Learning", box=box.ROUNDED)
        table.add_column("Factor", style="bold magenta")
        table.add_column("Value", justify="right")
        table.add_column("Description", style="dim")

        if self._three_factor:
            signal = self._three_factor.get_current_signal()

            # Factor 1: Eligibility
            table.add_row(
                "Eligibility",
                f"{signal.eligibility:.4f}",
                "Temporal credit assignment",
            )

            # Factor 2: Neuromodulator gate
            table.add_row(
                "Neuromod Gate",
                f"{signal.neuromod_gate:.4f}",
                "Should we learn now?",
            )

            # Factor 3: Dopamine surprise
            table.add_row(
                "DA Surprise",
                f"{signal.dopamine_surprise:+.4f}",
                "Prediction error magnitude",
            )

            table.add_section()

            # Effective learning rate
            effective_lr = signal.eligibility * signal.neuromod_gate * abs(signal.dopamine_surprise)
            table.add_row(
                "[bold]Effective LR[/bold]",
                f"[bold]{effective_lr:.4f}[/bold]",
                "Product of all factors",
            )

            # Learning direction
            direction = "+" if signal.dopamine_surprise > 0 else "-"
            direction_color = "green" if signal.dopamine_surprise > 0 else "red"
            table.add_row(
                "Direction",
                f"[{direction_color}]{direction}[/{direction_color}]",
                "Strengthen (+) or weaken (-)",
            )
        else:
            table.add_row("N/A", "0.0", "Not initialized")

        return table

    def show_three_factor(self) -> None:
        """Display three-factor learning state."""
        self.console.print(Panel(
            self._render_three_factor_table(),
            border_style="magenta",
        ))

        # Visual breakdown
        if self._three_factor:
            signal = self._three_factor.get_current_signal()

            breakdown = Text()
            breakdown.append("Learning Signal Breakdown:\n\n", style="bold")
            breakdown.append("  effective_lr = ")
            breakdown.append(f"eligibility({signal.eligibility:.3f})", style="cyan")
            breakdown.append(" x ")
            breakdown.append(f"gate({signal.neuromod_gate:.3f})", style="yellow")
            breakdown.append(" x ")
            breakdown.append(f"|surprise({signal.dopamine_surprise:+.3f})|", style="red")
            breakdown.append("\n\n")

            effective = signal.eligibility * signal.neuromod_gate * abs(signal.dopamine_surprise)
            breakdown.append(f"  = {effective:.4f}", style="bold green")

            self.console.print(Panel(breakdown, title="Formula", border_style="green"))

    # =========================================================================
    # FSRS Memory Scheduling
    # =========================================================================

    def _render_fsrs_stats(self) -> Table:
        """Render FSRS scheduling statistics."""
        table = Table(title="FSRS Memory Scheduling", box=box.ROUNDED)
        table.add_column("Metric", style="bold blue")
        table.add_column("Value", style="white")

        if self._memory_tracker:
            stats = self._memory_tracker.get_stats()

            table.add_row("Total Memories", str(stats.get("total_memories", 0)))
            table.add_row("Due Now", str(stats.get("due_now", 0)))
            table.add_row("Due Today", str(stats.get("due_today", 0)))
            table.add_row("Average Stability", f"{stats.get('avg_stability', 0.0):.2f} days")
            table.add_row("Average Difficulty", f"{stats.get('avg_difficulty', 0.0):.2f}")

            # Rating distribution
            table.add_section()
            table.add_row("[bold]Recent Ratings[/bold]", "")
            for rating, count in stats.get("rating_counts", {}).items():
                table.add_row(f"  {rating}", str(count))
        else:
            table.add_row("Status", "Not initialized")

        return table

    def _render_memory_queue(self, limit: int = 10) -> Table:
        """Render upcoming memory reviews."""
        table = Table(title=f"Review Queue (next {limit})", box=box.ROUNDED)
        table.add_column("Memory ID", style="cyan", width=10)
        table.add_column("Due", style="white")
        table.add_column("Stability", justify="right")
        table.add_column("Difficulty", justify="right")
        table.add_column("Reps", justify="right")

        if self._memory_tracker:
            queue = self._memory_tracker.get_due_memories(limit=limit)

            for memory_id, state in queue:
                due_delta = state.due - datetime.now()
                if due_delta.total_seconds() < 0:
                    due_str = "[red]Overdue[/red]"
                elif due_delta.total_seconds() < 3600:
                    due_str = f"[yellow]{int(due_delta.total_seconds() / 60)}m[/yellow]"
                elif due_delta.total_seconds() < 86400:
                    due_str = f"{int(due_delta.total_seconds() / 3600)}h"
                else:
                    due_str = f"{int(due_delta.days)}d"

                table.add_row(
                    str(memory_id)[:10],
                    due_str,
                    f"{state.stability:.1f}d",
                    f"{state.difficulty:.2f}",
                    str(state.reps),
                )
        else:
            table.add_row("N/A", "", "", "", "")

        return table

    def show_fsrs(self) -> None:
        """Display FSRS scheduling state."""
        layout = Layout()
        layout.split_row(
            Layout(name="stats", ratio=1),
            Layout(name="queue", ratio=1),
        )

        layout["stats"].update(Panel(self._render_fsrs_stats(), border_style="blue"))
        layout["queue"].update(Panel(self._render_memory_queue(), border_style="cyan"))

        self.console.print(layout)

    # =========================================================================
    # Causal Attribution
    # =========================================================================

    def _render_causal_graph(self) -> Tree:
        """Render causal graph as tree."""
        tree = Tree("[bold cyan]Causal Graph[/bold cyan]", guide_style="bright_blue")

        if self._causal_graph:
            nodes = self._causal_graph.nodes
            edges = self._causal_graph.edges

            for node_id, node_data in list(nodes.items())[:10]:
                node_branch = tree.add(f"[magenta]{node_id}[/magenta]")

                # Get outgoing edges
                outgoing = [e for e in edges if e.source == node_id]
                for edge in outgoing[:5]:
                    edge_style = "green" if edge.weight > 0.5 else "yellow"
                    node_branch.add(
                        f"[{edge_style}]--({edge.relation.name}: {edge.weight:.2f})--> "
                        f"[cyan]{edge.target}[/cyan]"
                    )
        else:
            tree.add("[dim]No causal graph[/dim]")

        return tree

    def _render_attribution_table(self) -> Table:
        """Render causal attribution results."""
        table = Table(title="Causal Attribution", box=box.ROUNDED)
        table.add_column("Memory", style="cyan")
        table.add_column("Attribution", justify="right")
        table.add_column("Confidence", justify="right")
        table.add_column("Path Length", justify="right")

        if self._causal_learner:
            attributions = self._causal_learner.get_recent_attributions(limit=10)

            for attr in attributions:
                # Color by attribution strength
                attr_val = attr.attribution_score
                color = "green" if attr_val > 0.7 else "yellow" if attr_val > 0.3 else "red"

                table.add_row(
                    str(attr.memory_id)[:10],
                    f"[{color}]{attr_val:.3f}[/{color}]",
                    f"{attr.confidence:.3f}",
                    str(attr.path_length),
                )
        else:
            table.add_row("N/A", "0.0", "0.0", "0")

        return table

    def show_causal(self) -> None:
        """Display causal discovery and attribution."""
        layout = Layout()
        layout.split_row(
            Layout(name="graph", ratio=1),
            Layout(name="attribution", ratio=1),
        )

        layout["graph"].update(Panel(self._render_causal_graph(), border_style="magenta"))
        layout["attribution"].update(Panel(self._render_attribution_table(), border_style="green"))

        self.console.print(layout)

    # =========================================================================
    # Learning Events
    # =========================================================================

    def _render_recent_events(self, limit: int = 10) -> Table:
        """Render recent learning events."""
        table = Table(title=f"Recent Events (last {limit})", box=box.ROUNDED)
        table.add_column("Time", style="dim", width=12)
        table.add_column("Type", style="bold")
        table.add_column("Outcome", width=10)
        table.add_column("RPE", justify="right", width=8)
        table.add_column("Memories", justify="right", width=8)

        for event in self._events[-limit:]:
            # Time
            time_str = event.timestamp.strftime("%H:%M:%S")

            # Outcome with color
            outcome_colors = {
                OutcomeType.SUCCESS: "green",
                OutcomeType.FAILURE: "red",
                OutcomeType.PARTIAL: "yellow",
                OutcomeType.NEUTRAL: "white",
            }
            oc = event.outcome.outcome_type
            outcome_str = f"[{outcome_colors.get(oc, 'white')}]{oc.name}[/{outcome_colors.get(oc, 'white')}]"

            # RPE
            rpe = event.outcome.reward_prediction_error
            rpe_color = "green" if rpe > 0 else "red" if rpe < 0 else "white"
            rpe_str = f"[{rpe_color}]{rpe:+.3f}[/{rpe_color}]"

            table.add_row(
                time_str,
                event.retrieval.memory_type.name,
                outcome_str,
                rpe_str,
                str(len(event.retrieval.retrieved_ids)),
            )

        return table

    def show_events(self) -> None:
        """Display recent learning events."""
        self.console.print(Panel(
            self._render_recent_events(),
            border_style="yellow",
        ))

    # =========================================================================
    # Simulation
    # =========================================================================

    def simulate_outcome(
        self,
        outcome_type: OutcomeType = OutcomeType.SUCCESS,
        n_memories: int = 3,
    ) -> dict:
        """
        Simulate a learning outcome.

        Args:
            outcome_type: Type of outcome
            n_memories: Number of memories involved

        Returns:
            Learning statistics
        """
        # Create mock retrieval event
        retrieval = RetrievalEvent(
            query="simulated query",
            retrieved_ids=[UUID(int=i) for i in range(n_memories)],
            scores=[0.8 - i * 0.1 for i in range(n_memories)],
            memory_type=MemoryType.EPISODIC,
            timestamp=datetime.now(),
        )

        # Create outcome event
        reward = {
            OutcomeType.SUCCESS: 1.0,
            OutcomeType.PARTIAL: 0.5,
            OutcomeType.FAILURE: -1.0,
            OutcomeType.NEUTRAL: 0.0,
        }.get(outcome_type, 0.0)

        outcome = OutcomeEvent(
            outcome_type=outcome_type,
            reward=reward,
            timestamp=datetime.now(),
        )

        # Create experience
        experience = Experience(retrieval=retrieval, outcome=outcome)
        self._events.append(experience)

        # Process through neuromodulators
        if self._orchestra:
            self._orchestra.process_outcome(experience)

        # Update eligibility
        if self._eligibility:
            self._eligibility.update(reward)

        # Record snapshot
        self._record_snapshot()

        return {
            "outcome_type": outcome_type.name,
            "reward": reward,
            "rpe": self._dopamine.rpe if self._dopamine else 0.0,
            "learning_gate": self._three_factor.compute_multiplier() if self._three_factor else 1.0,
        }

    # =========================================================================
    # Interactive Mode
    # =========================================================================

    def show_dashboard(self) -> None:
        """Display full learning dashboard."""
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
            Layout(name="neuromod", ratio=1),
            Layout(name="three_factor", ratio=1),
        )

        layout["right"].split_column(
            Layout(name="eligibility", ratio=1),
            Layout(name="events", ratio=1),
        )

        # Header
        header_text = Text("T4DM Learning Inspector", style="bold cyan")
        layout["header"].update(Panel(header_text, border_style="cyan"))

        # Panels
        layout["neuromod"].update(Panel(self._render_neuromodulator_state(), border_style="magenta"))
        layout["three_factor"].update(Panel(self._render_three_factor_table(), border_style="green"))
        layout["eligibility"].update(Panel(self._render_eligibility_table(), border_style="cyan"))
        layout["events"].update(Panel(self._render_recent_events(5), border_style="yellow"))

        # Footer
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        footer_text = Text(
            f"Session: {self.session_id or 'ALL'} | "
            f"Events: {len(self._events)} | "
            f"Updated: {now}",
            style="dim",
        )
        layout["footer"].update(Panel(footer_text, border_style="dim"))

        self.console.print(layout)

    def interactive(self) -> None:
        """
        Run interactive learning inspector session.

        Provides menu-driven interface for inspecting learning dynamics.
        """
        self.console.print(Panel(
            "[bold cyan]T4DM Learning Inspector[/bold cyan]\n"
            "Interactive learning dynamics exploration interface",
            border_style="cyan",
        ))

        while True:
            self.console.print("\n[bold magenta]Options:[/bold magenta]")
            self.console.print("1. View Neuromodulators")
            self.console.print("2. View Eligibility Traces")
            self.console.print("3. View Three-Factor Learning")
            self.console.print("4. View FSRS Scheduling")
            self.console.print("5. View Causal Attribution")
            self.console.print("6. View Recent Events")
            self.console.print("7. Simulate Success")
            self.console.print("8. Simulate Failure")
            self.console.print("9. Full Dashboard")
            self.console.print("0. Exit")

            choice = Prompt.ask(
                "\n[bold cyan]Select option[/bold cyan]",
                choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                default="0",
            )

            if choice == "1":
                self.show_neuromodulators()
            elif choice == "2":
                self.show_eligibility()
            elif choice == "3":
                self.show_three_factor()
            elif choice == "4":
                self.show_fsrs()
            elif choice == "5":
                self.show_causal()
            elif choice == "6":
                self.show_events()
            elif choice == "7":
                result = self.simulate_outcome(OutcomeType.SUCCESS)
                self.console.print(f"[green]Simulated SUCCESS: RPE={result['rpe']:+.3f}[/green]")
            elif choice == "8":
                result = self.simulate_outcome(OutcomeType.FAILURE)
                self.console.print(f"[red]Simulated FAILURE: RPE={result['rpe']:+.3f}[/red]")
            elif choice == "9":
                self.show_dashboard()
            elif choice == "0":
                self.console.print("\n[bold green]Goodbye![/bold green]")
                break

    def get_stats(self) -> dict:
        """Get learning inspector statistics."""
        return {
            "initialized": self._initialized,
            "session_id": self.session_id,
            "history_size": len(self._history),
            "event_count": len(self._events),
            "neuromodulator_state": {
                "dopamine_rpe": self._dopamine.rpe if self._dopamine else None,
                "serotonin_credit": self._serotonin.credit if self._serotonin else None,
                "norepinephrine_arousal": self._norepinephrine.arousal if self._norepinephrine else None,
                "acetylcholine_mode": self._acetylcholine.mode.name if self._acetylcholine else None,
            },
        }


def main():
    """CLI entry point for learning inspector."""
    import sys

    session_id = sys.argv[1] if len(sys.argv) > 1 else None

    inspector = LearningInspector(session_id=session_id)
    inspector.interactive()


if __name__ == "__main__":
    main()
