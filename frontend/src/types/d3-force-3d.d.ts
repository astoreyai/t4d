/**
 * Type declarations for d3-force-3d
 * This package extends d3-force with 3D capabilities
 */

declare module 'd3-force-3d' {
  export interface SimulationNodeDatum {
    index?: number;
    x?: number;
    y?: number;
    z?: number;
    vx?: number;
    vy?: number;
    vz?: number;
    fx?: number | null;
    fy?: number | null;
    fz?: number | null;
  }

  export interface SimulationLinkDatum<NodeDatum extends SimulationNodeDatum> {
    source: NodeDatum | string | number;
    target: NodeDatum | string | number;
    index?: number;
  }

  export interface Simulation<
    NodeDatum extends SimulationNodeDatum,
    LinkDatum extends SimulationLinkDatum<NodeDatum> | undefined
  > {
    restart(): this;
    stop(): this;
    tick(iterations?: number): this;
    nodes(): NodeDatum[];
    nodes(nodes: NodeDatum[]): this;
    alpha(): number;
    alpha(alpha: number): this;
    alphaMin(): number;
    alphaMin(min: number): this;
    alphaDecay(): number;
    alphaDecay(decay: number): this;
    alphaTarget(): number;
    alphaTarget(target: number): this;
    velocityDecay(): number;
    velocityDecay(decay: number): this;
    force<F extends Force<NodeDatum, LinkDatum>>(name: string): F | undefined;
    force(name: string, force: Force<NodeDatum, LinkDatum> | null): this;
    find(x: number, y: number, z?: number, radius?: number): NodeDatum | undefined;
    randomSource(): () => number;
    randomSource(source: () => number): this;
    on(typenames: string): ((this: this) => void) | undefined;
    on(typenames: string, listener: ((this: this) => void) | null): this;
  }

  export interface Force<
    NodeDatum extends SimulationNodeDatum,
    LinkDatum extends SimulationLinkDatum<NodeDatum> | undefined
  > {
    (alpha: number): void;
    initialize?(nodes: NodeDatum[], random: () => number): void;
  }

  export interface ForceLink<
    NodeDatum extends SimulationNodeDatum,
    LinkDatum extends SimulationLinkDatum<NodeDatum>
  > extends Force<NodeDatum, LinkDatum> {
    links(): LinkDatum[];
    links(links: LinkDatum[]): this;
    id(): (node: NodeDatum, i: number, nodes: NodeDatum[]) => string | number;
    id(id: (node: NodeDatum, i: number, nodes: NodeDatum[]) => string | number): this;
    iterations(): number;
    iterations(iterations: number): this;
    strength(): (link: LinkDatum, i: number, links: LinkDatum[]) => number;
    strength(strength: number | ((link: LinkDatum, i: number, links: LinkDatum[]) => number)): this;
    distance(): (link: LinkDatum, i: number, links: LinkDatum[]) => number;
    distance(distance: number | ((link: LinkDatum, i: number, links: LinkDatum[]) => number)): this;
  }

  export interface ForceManyBody<NodeDatum extends SimulationNodeDatum>
    extends Force<NodeDatum, undefined> {
    strength(): (node: NodeDatum, i: number, nodes: NodeDatum[]) => number;
    strength(strength: number | ((node: NodeDatum, i: number, nodes: NodeDatum[]) => number)): this;
    theta(): number;
    theta(theta: number): this;
    distanceMin(): number;
    distanceMin(distance: number): this;
    distanceMax(): number;
    distanceMax(distance: number): this;
  }

  export interface ForceCenter<NodeDatum extends SimulationNodeDatum>
    extends Force<NodeDatum, undefined> {
    x(): number;
    x(x: number): this;
    y(): number;
    y(y: number): this;
    z(): number;
    z(z: number): this;
    strength(): number;
    strength(strength: number): this;
  }

  export interface ForceCollide<NodeDatum extends SimulationNodeDatum>
    extends Force<NodeDatum, undefined> {
    radius(): (node: NodeDatum, i: number, nodes: NodeDatum[]) => number;
    radius(radius: number | ((node: NodeDatum, i: number, nodes: NodeDatum[]) => number)): this;
    strength(): number;
    strength(strength: number): this;
    iterations(): number;
    iterations(iterations: number): this;
  }

  export function forceSimulation<NodeDatum extends SimulationNodeDatum>(
    nodes?: NodeDatum[],
    numDimensions?: number
  ): Simulation<NodeDatum, undefined>;

  export function forceLink<
    NodeDatum extends SimulationNodeDatum,
    LinkDatum extends SimulationLinkDatum<NodeDatum>
  >(links?: LinkDatum[]): ForceLink<NodeDatum, LinkDatum>;

  export function forceManyBody<NodeDatum extends SimulationNodeDatum>(): ForceManyBody<NodeDatum>;

  export function forceCenter<NodeDatum extends SimulationNodeDatum>(
    x?: number,
    y?: number,
    z?: number
  ): ForceCenter<NodeDatum>;

  export function forceCollide<NodeDatum extends SimulationNodeDatum>(
    radius?: number | ((node: NodeDatum, i: number, nodes: NodeDatum[]) => number)
  ): ForceCollide<NodeDatum>;

  export function forceX<NodeDatum extends SimulationNodeDatum>(
    x?: number | ((node: NodeDatum, i: number, nodes: NodeDatum[]) => number)
  ): Force<NodeDatum, undefined>;

  export function forceY<NodeDatum extends SimulationNodeDatum>(
    y?: number | ((node: NodeDatum, i: number, nodes: NodeDatum[]) => number)
  ): Force<NodeDatum, undefined>;

  export function forceZ<NodeDatum extends SimulationNodeDatum>(
    z?: number | ((node: NodeDatum, i: number, nodes: NodeDatum[]) => number)
  ): Force<NodeDatum, undefined>;

  export function forceRadial<NodeDatum extends SimulationNodeDatum>(
    radius?: number | ((node: NodeDatum, i: number, nodes: NodeDatum[]) => number),
    x?: number,
    y?: number,
    z?: number
  ): Force<NodeDatum, undefined>;
}
