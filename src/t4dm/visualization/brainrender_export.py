"""
BrainRender Export Module (P5-01).

Exports T4DM connectome and activity data for 3D visualization
using BrainRender (https://brainrender.info/).

Features:
- Connectome structure export (regions + connections)
- Activity heatmap generation
- Pathway highlighting
- Animation-ready data export

Note: BrainRender is an optional dependency. This module provides
export functionality that can be used with BrainRender when installed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Check if BrainRender is available
BRAINRENDER_AVAILABLE = False
try:
    import brainrender
    from brainrender import Scene
    from brainrender.actors import Points, Line
    BRAINRENDER_AVAILABLE = True
    logger.info("BrainRender available for 3D visualization")
except ImportError:
    brainrender = None
    Scene = None
    Points = None
    Line = None
    logger.info("BrainRender not installed - export to JSON only")


@dataclass
class RegionData:
    """Data for a brain region."""
    name: str
    coordinates: tuple[float, float, float]
    region_type: str
    activity: float = 0.0
    color: str = "#808080"
    size: float = 1.0


@dataclass
class ConnectionData:
    """Data for a connection between regions."""
    source: str
    target: str
    weight: float
    nt_type: str
    color: str = "#404040"


@dataclass
class BrainRenderExport:
    """Complete export data for BrainRender visualization."""
    regions: list[RegionData]
    connections: list[ConnectionData]
    metadata: dict[str, Any]


class BrainRenderExporter:
    """
    Exports T4DM data for BrainRender 3D visualization.

    Supports:
    1. Connectome structure visualization
    2. Activity-based coloring
    3. Pathway highlighting
    4. JSON export for external rendering
    """

    # NT type to color mapping
    NT_COLORS = {
        "dopamine": "#FF6B6B",      # Red
        "serotonin": "#4ECDC4",     # Teal
        "norepinephrine": "#45B7D1", # Blue
        "acetylcholine": "#96CEB4",  # Green
        "glutamate": "#FFEAA7",      # Yellow
        "gaba": "#DDA0DD",           # Purple
    }

    # Region type to color mapping
    REGION_COLORS = {
        "CORTICAL": "#6C5CE7",       # Purple
        "SUBCORTICAL": "#00B894",    # Green
        "LIMBIC": "#E17055",         # Orange
        "BRAINSTEM": "#0984E3",      # Blue
        "CEREBELLAR": "#FDCB6E",     # Yellow
    }

    def __init__(self, scale_factor: float = 1.0):
        """
        Initialize exporter.

        Args:
            scale_factor: Scale factor for coordinates (default: 1.0)
        """
        self.scale_factor = scale_factor

    def export_connectome(
        self,
        connectome,  # Connectome from nca.connectome
        activity: dict[str, float] | None = None,
        highlight_pathway: str | None = None,
    ) -> BrainRenderExport:
        """
        Export connectome for BrainRender visualization.

        Args:
            connectome: Connectome instance from nca.connectome
            activity: Optional dict of region_name -> activity level [0,1]
            highlight_pathway: Optional NT type to highlight

        Returns:
            BrainRenderExport with all visualization data
        """
        from t4dm.nca.connectome import NTSystem

        regions = []
        connections = []

        # Export regions
        for name, region in connectome.regions.items():
            # Scale coordinates
            coords = tuple(c * self.scale_factor for c in region.coordinates)

            # Determine color based on activity or region type
            if activity and name in activity:
                # Activity-based coloring (red = high, blue = low)
                act = activity[name]
                color = self._activity_to_color(act)
                size = 1.0 + act * 2.0  # Larger for more active
            else:
                color = self.REGION_COLORS.get(region.region_type.name, "#808080")
                size = 1.0

            regions.append(RegionData(
                name=name,
                coordinates=coords,
                region_type=region.region_type.name,
                activity=activity.get(name, 0.0) if activity else 0.0,
                color=color,
                size=size,
            ))

        # Export connections
        region_names = list(connectome.regions.keys())

        for nt, matrix in connectome._connectivity.items():
            nt_name = nt.value if hasattr(nt, 'value') else str(nt)
            nt_color = self.NT_COLORS.get(nt_name, "#404040")

            # Highlight specific pathway
            if highlight_pathway and nt_name != highlight_pathway:
                nt_color = "#202020"  # Dim non-highlighted

            for i, src in enumerate(region_names):
                for j, tgt in enumerate(region_names):
                    weight = matrix[i, j]
                    if weight > 0.01:  # Threshold
                        connections.append(ConnectionData(
                            source=src,
                            target=tgt,
                            weight=float(weight),
                            nt_type=nt_name,
                            color=nt_color,
                        ))

        metadata = {
            "num_regions": len(regions),
            "num_connections": len(connections),
            "scale_factor": self.scale_factor,
            "highlight_pathway": highlight_pathway,
        }

        return BrainRenderExport(
            regions=regions,
            connections=connections,
            metadata=metadata,
        )

    def export_to_json(
        self,
        export_data: BrainRenderExport,
        output_path: str | Path,
    ) -> Path:
        """
        Export visualization data to JSON file.

        Args:
            export_data: BrainRenderExport data
            output_path: Output file path

        Returns:
            Path to written file
        """
        output_path = Path(output_path)

        # Convert to serializable format
        data = {
            "regions": [asdict(r) for r in export_data.regions],
            "connections": [asdict(c) for c in export_data.connections],
            "metadata": export_data.metadata,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported BrainRender data to {output_path}")
        return output_path

    def export_to_html(
        self,
        export_data: BrainRenderExport,
        output_path: str | Path,
        title: str = "T4DM Connectome",
    ) -> Path:
        """
        Export interactive HTML visualization using Three.js.

        Args:
            export_data: BrainRenderExport data
            output_path: Output HTML file path
            title: Page title

        Returns:
            Path to written file
        """
        output_path = Path(output_path)

        # Generate HTML with embedded Three.js visualization
        html = self._generate_threejs_html(export_data, title)

        with open(output_path, "w") as f:
            f.write(html)

        logger.info(f"Exported HTML visualization to {output_path}")
        return output_path

    def render_brainrender(
        self,
        export_data: BrainRenderExport,
        output_path: str | Path | None = None,
        screenshot: bool = True,
    ) -> Any:
        """
        Render using BrainRender (if available).

        Args:
            export_data: BrainRenderExport data
            output_path: Optional output path for screenshot
            screenshot: Whether to take screenshot

        Returns:
            BrainRender Scene object (or None if not available)
        """
        if not BRAINRENDER_AVAILABLE:
            logger.warning("BrainRender not installed - use export_to_html instead")
            return None

        # Create scene
        scene = Scene(title="T4DM Connectome")

        # Add regions as points
        coords = np.array([r.coordinates for r in export_data.regions])
        colors = [r.color for r in export_data.regions]
        sizes = [r.size * 100 for r in export_data.regions]

        points = Points(coords, colors=colors, radius=sizes)
        scene.add(points)

        # Add connections as lines
        region_coords = {r.name: r.coordinates for r in export_data.regions}

        for conn in export_data.connections:
            if conn.source in region_coords and conn.target in region_coords:
                start = region_coords[conn.source]
                end = region_coords[conn.target]
                line = Line(
                    [start, end],
                    color=conn.color,
                    linewidth=conn.weight * 3,
                )
                scene.add(line)

        if screenshot and output_path:
            scene.screenshot(name=str(output_path))

        return scene

    def _activity_to_color(self, activity: float) -> str:
        """Convert activity level [0,1] to color (blue→red)."""
        # Clamp to [0, 1]
        activity = max(0.0, min(1.0, activity))

        # Interpolate from blue (low) to red (high)
        r = int(activity * 255)
        b = int((1 - activity) * 255)
        g = int((1 - abs(activity - 0.5) * 2) * 128)

        return f"#{r:02x}{g:02x}{b:02x}"

    def _generate_threejs_html(
        self,
        export_data: BrainRenderExport,
        title: str,
    ) -> str:
        """Generate Three.js HTML visualization."""
        # Convert data to JSON for embedding
        regions_json = json.dumps([asdict(r) for r in export_data.regions])
        connections_json = json.dumps([asdict(c) for c in export_data.connections])

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ margin: 0; overflow: hidden; }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            font-family: Arial, sans-serif;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 5px;
        }}
        #legend {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            color: white;
            font-family: Arial, sans-serif;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 5px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 10px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div id="info">
        <h3>{title}</h3>
        <p>Regions: {len(export_data.regions)}</p>
        <p>Connections: {len(export_data.connections)}</p>
        <p>Drag to rotate, scroll to zoom</p>
    </div>
    <div id="legend">
        <h4>Region Types</h4>
        <div class="legend-item">
            <div class="legend-color" style="background: #6C5CE7"></div>
            <span>Cortical</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #00B894"></div>
            <span>Subcortical</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #E17055"></div>
            <span>Limbic</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #0984E3"></div>
            <span>Brainstem</span>
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        const regions = {regions_json};
        const connections = {connections_json};

        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);

        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.z = 100;

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        // Add ambient light
        const ambientLight = new THREE.AmbientLight(0x404040);
        scene.add(ambientLight);

        const pointLight = new THREE.PointLight(0xffffff, 1, 200);
        pointLight.position.set(50, 50, 50);
        scene.add(pointLight);

        // Add regions as spheres
        regions.forEach(region => {{
            const geometry = new THREE.SphereGeometry(region.size * 2, 16, 16);
            const material = new THREE.MeshPhongMaterial({{
                color: region.color,
                emissive: region.color,
                emissiveIntensity: 0.3,
            }});
            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(...region.coordinates);
            sphere.userData = {{ name: region.name, type: region.region_type }};
            scene.add(sphere);
        }});

        // Add connections as lines
        const lineMaterial = new THREE.LineBasicMaterial({{
            color: 0x404040,
            transparent: true,
            opacity: 0.3,
        }});

        const regionCoords = {{}};
        regions.forEach(r => {{ regionCoords[r.name] = r.coordinates; }});

        connections.forEach(conn => {{
            if (regionCoords[conn.source] && regionCoords[conn.target]) {{
                const points = [
                    new THREE.Vector3(...regionCoords[conn.source]),
                    new THREE.Vector3(...regionCoords[conn.target]),
                ];
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const material = new THREE.LineBasicMaterial({{
                    color: conn.color,
                    transparent: true,
                    opacity: Math.min(0.8, conn.weight + 0.2),
                }});
                const line = new THREE.Line(geometry, material);
                scene.add(line);
            }}
        }});

        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();

        // Handle resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>"""
        return html


def export_connectome_visualization(
    connectome,
    output_dir: str | Path,
    activity: dict[str, float] | None = None,
    formats: list[str] = ["json", "html"],
) -> dict[str, Path]:
    """
    Convenience function to export connectome in multiple formats.

    Args:
        connectome: Connectome instance
        output_dir: Output directory
        activity: Optional activity data
        formats: List of formats to export ("json", "html", "brainrender")

    Returns:
        Dict of format -> output path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exporter = BrainRenderExporter()
    export_data = exporter.export_connectome(connectome, activity=activity)

    outputs = {}

    if "json" in formats:
        path = exporter.export_to_json(export_data, output_dir / "connectome.json")
        outputs["json"] = path

    if "html" in formats:
        path = exporter.export_to_html(export_data, output_dir / "connectome.html")
        outputs["html"] = path

    if "brainrender" in formats and BRAINRENDER_AVAILABLE:
        scene = exporter.render_brainrender(
            export_data,
            output_path=output_dir / "connectome.png",
        )
        if scene:
            outputs["brainrender"] = output_dir / "connectome.png"

    return outputs


__all__ = [
    "BrainRenderExporter",
    "BrainRenderExport",
    "RegionData",
    "ConnectionData",
    "export_connectome_visualization",
    "BRAINRENDER_AVAILABLE",
]
