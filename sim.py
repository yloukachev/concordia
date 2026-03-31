"""
Concordia Simulation: The Settlement
=====================================
A post-collapse settlement where agents face survival pressure,
form social bonds, trade resources, and navigate power dynamics.

Features:
  - Survival pressure (scarcity, needs, danger)
  - Social dynamics (alliances, conflict, trust)
  - Commercial activity (trade, resource management)
  - Visualization (social graph, timeline, agent states)
  - God Mode (inject events, control agents, interview them)

Usage:
  export ANTHROPIC_API_KEY=your-key-here
  python sim.py
"""

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from concordia.contrib import language_models as language_model_utils
import concordia.prefabs.entity as entity_prefabs
import concordia.prefabs.game_master as game_master_prefabs
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions
import sentence_transformers

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_TYPE = 'claude_code'
MODEL_NAME = 'claude-code'  # Uses your Claude Code subscription, no API key
MAX_STEPS = 10  # Increase for longer simulations (each step = 1 agent action)

# The world
PREMISE = textwrap.dedent("""\
    The year is 2031. Six months ago, a cascading infrastructure failure
    knocked out power grids across the western seaboard. The old world is
    gone. Small settlements have formed from the survivors.

    Dusthaven is one such settlement — 5 people camped around an
    abandoned gas station at a desert crossroads. Water is scarce. Food
    comes from hunting and a small garden. A trader caravan passes
    through every few weeks, but they demand high prices.

    Winter is approaching. The settlement has enough food for maybe 3
    weeks. There are rumors of a larger, fortified community called
    "Haven" two days' walk north, but also rumors of raiders on that
    road.

    Resources at the settlement:
    - Water: a well that produces ~10 gallons/day (barely enough for 5)
    - Food: canned goods (running low), a small vegetable garden, hunting
    - Shelter: the gas station building, some tents
    - Tools: basic hand tools, one hunting rifle with limited ammo
    - Medicine: a small first aid kit, nearly depleted
    - Trade goods: some gasoline from the station's underground tanks

    The settlement has no formal leader. Decisions are made by whoever
    speaks loudest or acts first. Tensions are rising.
""")

SHARED_MEMORIES = [
    'Dusthaven is a small settlement of 5 survivors at a desert crossroads.',
    'The infrastructure collapse happened 6 months ago. No power grid, no government.',
    'Water comes from a well that barely produces enough for everyone.',
    'Food supplies will run out in about 3 weeks unless something changes.',
    'Winter is approaching and nights are getting dangerously cold.',
    'A trader caravan visited last week demanding 5 gallons of gasoline for basic medicine.',
    'There are rumors of a fortified community called Haven two days north.',
    'Raiders have been spotted on the northern road.',
    'The settlement has one hunting rifle with 12 bullets remaining.',
    'There is no formal leadership structure — decisions are contested.',
]

# Agent definitions
AGENTS = [
    {
        'name': 'Marcus Cole',
        'goal': 'Ensure the survival of the settlement by establishing order and preparing for winter',
        'context': (
            'Marcus Cole is a 45-year-old former Army sergeant. He is '
            'disciplined, pragmatic, and believes strong leadership is essential '
            'for survival. He keeps the rifle and considers himself the '
            "settlement's protector. He distrusts outsiders and thinks the group "
            'should fortify rather than seek Haven. He has a quiet temper but '
            'can be authoritarian when stressed. He rations food strictly.'
        ),
    },
    {
        'name': 'Elena Vasquez',
        'goal': 'Keep everyone alive and healthy, and convince the group to seek Haven before winter',
        'context': (
            'Elena Vasquez is a 34-year-old former ER nurse. She is '
            'compassionate, resourceful, and the only one with medical '
            'knowledge. She believes staying at Dusthaven through winter is '
            'a death sentence and wants to lead an expedition to Haven. '
            'She worries about Marcus consolidating power around the rifle. '
            'She trades medical care for influence and goodwill.'
        ),
    },
    {
        'name': 'Dex Rourke',
        'goal': 'Accumulate resources and leverage for personal advantage, keep all options open',
        'context': (
            'Dex Rourke is a 28-year-old former used car salesman. He is '
            'charming, opportunistic, and a natural negotiator. He secretly '
            'has a stash of canned food hidden outside camp that nobody '
            'knows about. He plays both sides — supporting Marcus to his '
            'face while encouraging Elena behind his back. He wants to be '
            'indispensable to whoever ends up in charge. He is the main '
            'contact with the trader caravans.'
        ),
    },
    {
        'name': 'Priya Okafor',
        'goal': 'Protect the community garden and build sustainable food sources so the group can survive long-term',
        'context': (
            'Priya Okafor is a 52-year-old former agricultural scientist. '
            'She is patient, knowledgeable, and quietly stubborn. She '
            'manages the vegetable garden and believes sustainable farming '
            'is the only real path to survival. She resents that Marcus '
            "controls the rifle while she controls the food. She doesn't "
            'trust Dex at all. She would consider Haven only if the group '
            'could bring seeds and establish farming there.'
        ),
    },
    {
        'name': 'Kai Tanaka',
        'goal': 'Find purpose and belonging, prove worth to the group through scouting and resourcefulness',
        'context': (
            'Kai Tanaka is a 19-year-old college student who was road-tripping '
            'when the collapse happened. The youngest in the group, Kai is '
            'energetic, idealistic, and eager to prove themselves. They are '
            'the best scout and forager — quick, quiet, and observant. They '
            'look up to Elena and are wary of Marcus. They found raider '
            'tracks 2 days ago but only told Elena, fearing Marcus would '
            'use it as an excuse to lock down the camp.'
        ),
    },
]

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _parse_log(raw_log):
    """Parse Concordia's raw_log format into (actor, action_text, event_text) per step."""
    agent_names = [a['name'] for a in AGENTS]
    steps = []
    for entry in raw_log:
        if not isinstance(entry, dict):
            continue
        # Find which agent acted this step
        actor = None
        action_text = ''
        event_text = ''
        for key, val in entry.items():
            if key.startswith('Entity ['):
                actor = key[len('Entity ['):-1]
                if isinstance(val, dict) and '__act__' in val:
                    act = val['__act__']
                    if isinstance(act, dict):
                        action_text = act.get('Summary', act.get('Value', ''))
                    else:
                        action_text = str(act)
            elif key not in ('Step', 'Summary') and isinstance(key, str):
                # Event resolution keys contain the narrative
                event_text = key
        if actor:
            steps.append({
                'step': entry.get('Step', len(steps)),
                'actor': actor,
                'action': action_text,
                'event': event_text,
            })
    return steps


def visualize_simulation(results_log, raw_log, output_dir='sim_output'):
    """Generate visualizations from the simulation results."""
    os.makedirs(output_dir, exist_ok=True)

    agent_names = [a['name'] for a in AGENTS]
    agent_colors = {
        'Marcus Cole': '#e74c3c',
        'Elena Vasquez': '#3498db',
        'Dex Rourke': '#f39c12',
        'Priya Okafor': '#27ae60',
        'Kai Tanaka': '#9b59b6',
    }
    short_names = [name.split()[0] for name in agent_names]

    steps = _parse_log(raw_log)
    if not steps:
        print('  No parsed steps — skipping visualizations.')
        return

    # --- 1. Action Timeline ---
    fig, ax = plt.subplots(figsize=(14, 6))
    y_pos = {name: i for i, name in enumerate(agent_names)}
    for i, s in enumerate(steps):
        if s['actor'] in agent_names:
            ax.scatter(i, y_pos[s['actor']],
                      c=agent_colors.get(s['actor'], 'gray'),
                      s=200, zorder=3, edgecolors='black', linewidth=1)
            # Add short action label
            label = s['action'][:40] + '...' if len(s['action']) > 40 else s['action']
            ax.annotate(label, (i, y_pos[s['actor']]),
                       textcoords="offset points", xytext=(10, 5),
                       fontsize=7, alpha=0.7, wrap=True)

    ax.set_yticks(range(len(agent_names)))
    ax.set_yticklabels(agent_names, fontsize=11)
    ax.set_xlabel('Simulation Step', fontsize=12)
    ax.set_title('Agent Activity Timeline', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(-0.5, len(steps) - 0.5)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/timeline.png', dpi=150)
    plt.close()

    # --- 2. Interaction Matrix (who mentions whom in events) ---
    n = len(agent_names)
    interaction_matrix = np.zeros((n, n))

    for s in steps:
        actor = s['actor']
        text = s['action'] + ' ' + s['event']
        if actor in agent_names:
            actor_idx = agent_names.index(actor)
            for j, other in enumerate(agent_names):
                if other != actor:
                    # Match full name or first name
                    first_name = other.split()[0]
                    if other in text or first_name in text:
                        interaction_matrix[actor_idx][j] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(interaction_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(short_names, fontsize=11)
    ax.set_title('Interaction Heatmap\n(who mentions whom in actions/events)',
                fontsize=14, fontweight='bold')

    for i in range(n):
        for j in range(n):
            val = int(interaction_matrix[i][j])
            if val > 0:
                ax.text(j, i, str(val), ha='center', va='center',
                       fontsize=13, fontweight='bold',
                       color='white' if val > 2 else 'black')

    plt.colorbar(im, ax=ax, label='Mentions')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/interactions.png', dpi=150)
    plt.close()

    # --- 3. Social Network Graph ---
    fig, ax = plt.subplots(figsize=(10, 10))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    radius = 3.0
    positions = {name: (radius * np.cos(a), radius * np.sin(a))
                 for name, a in zip(agent_names, angles)}

    # Symmetric interaction weights for edges
    max_weight = 1
    for i in range(n):
        for j in range(i + 1, n):
            w = interaction_matrix[i][j] + interaction_matrix[j][i]
            if w > max_weight:
                max_weight = w

    for i, name_i in enumerate(agent_names):
        for j, name_j in enumerate(agent_names):
            weight = interaction_matrix[i][j] + interaction_matrix[j][i]
            if weight > 0 and i < j:
                x = [positions[name_i][0], positions[name_j][0]]
                y = [positions[name_i][1], positions[name_j][1]]
                norm_w = weight / max_weight
                ax.plot(x, y, color='#555555',
                       alpha=0.2 + 0.6 * norm_w,
                       linewidth=1 + 5 * norm_w)

    # Draw nodes
    for name in agent_names:
        x, y = positions[name]
        # Size by how many actions this agent took
        action_count = sum(1 for s in steps if s['actor'] == name)
        node_size = 0.4 + 0.15 * action_count
        circle = plt.Circle((x, y), node_size,
                           color=agent_colors.get(name, 'gray'),
                           ec='black', linewidth=2, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y - node_size - 0.3, name,
               ha='center', va='top', fontsize=11, fontweight='bold')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title('Social Network\n(edge thickness = interaction frequency, node size = actions taken)',
                fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/social_network.png', dpi=150)
    plt.close()

    # --- 4. Narrative Summary ---
    with open(f'{output_dir}/narrative.txt', 'w') as f:
        f.write('THE SETTLEMENT — Simulation Narrative\n')
        f.write('=' * 50 + '\n\n')
        for s in steps:
            f.write(f'[Step {s["step"]}] {s["actor"]}:\n')
            # Extract just the event resolution (after "Event:")
            event = s['event']
            if 'Event:' in event:
                event = event.split('Event:', 1)[1].strip()
            f.write(textwrap.fill(event[:500], width=76, initial_indent='  ',
                                 subsequent_indent='  '))
            f.write('\n\n')

    print(f'\nVisualizations saved to {output_dir}/')
    print(f'  - timeline.png')
    print(f'  - interactions.png')
    print(f'  - social_network.png')
    print(f'  - narrative.txt')


# ---------------------------------------------------------------------------
# God Mode — interactive control
# ---------------------------------------------------------------------------

def god_mode(sim):
    """Interactive god mode for the simulation."""
    print('\n' + '=' * 60)
    print('  GOD MODE — You control the world')
    print('=' * 60)
    print(textwrap.dedent("""\
    Commands:
      observe <name>     — See what an agent is thinking
      inject <text>      — Inject a world event (all agents see it)
      whisper <name> <msg> — Whisper something only one agent hears
      step [n]           — Run n simulation steps (default: 1)
      status             — Show simulation state
      quit               — Exit god mode and finalize
    """))

    raw_log = []
    total_steps = 0

    while True:
        try:
            cmd = input('\n[GOD] > ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nExiting god mode.')
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()

        if action == 'quit':
            break

        elif action == 'status':
            print(f'  Total steps run: {total_steps}')
            print(f'  Agents: {[e.name for e in sim.entities]}')
            print(f'  Log entries: {len(raw_log)}')

        elif action == 'step':
            n = int(parts[1]) if len(parts) > 1 else 1
            print(f'  Running {n} step(s)...')
            results = sim.play(max_steps=n, raw_log=raw_log)
            total_steps += n
            # Print the latest events
            for entry in raw_log[-n:]:
                name = entry.get('entity_name', '???')
                action_text = entry.get('action', entry.get('observation', ''))
                if action_text:
                    print(f'  [{name}] {action_text}')

        elif action == 'observe':
            if len(parts) < 2:
                print('  Usage: observe <agent name>')
                continue
            target = parts[1]
            found = False
            for entity in sim.entities:
                if target.lower() in entity.name.lower():
                    print(f'\n  --- Observing {entity.name} ---')
                    try:
                        memory = entity.get_component('__memory__')
                        state = memory.get_state()
                        memories = state.get('memory_bank', {}).get('entries', [])
                        if memories:
                            print(f'  Recent memories ({min(5, len(memories))} of {len(memories)}):')
                            for m in memories[-5:]:
                                text = m if isinstance(m, str) else m.get('text', str(m))
                                print(f'    - {text[:120]}')
                        else:
                            print('  (no memories found in expected format)')
                    except Exception as e:
                        print(f'  Could not read memories: {e}')
                    found = True
                    break
            if not found:
                print(f'  Agent "{target}" not found.')

        elif action == 'inject':
            if len(parts) < 2:
                print('  Usage: inject <event description>')
                continue
            event = parts[1]
            print(f'  Injecting event: {event}')
            for entity in sim.entities:
                try:
                    entity.observe(event)
                except Exception:
                    pass  # GM entities may not support observe
            print('  Event injected to all agents.')

        elif action == 'whisper':
            subparts = cmd.split(maxsplit=2)
            if len(subparts) < 3:
                print('  Usage: whisper <agent name> <message>')
                continue
            target, msg = subparts[1], subparts[2]
            found = False
            for entity in sim.entities:
                if target.lower() in entity.name.lower():
                    entity.observe(msg)
                    print(f'  Whispered to {entity.name}: {msg}')
                    found = True
                    break
            if not found:
                print(f'  Agent "{target}" not found.')

        else:
            print(f'  Unknown command: {action}')

    return raw_log, total_steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Concordia Settlement Simulation')
    parser.add_argument('--mode', choices=['auto', 'god', 'both'],
                        default=None, help='Run mode (skip interactive menu)')
    parser.add_argument('--steps', type=int, default=MAX_STEPS,
                        help=f'Number of simulation steps (default: {MAX_STEPS})')
    args = parser.parse_args()

    print('=' * 60)
    print('  CONCORDIA SIMULATION: THE SETTLEMENT')
    print('=' * 60)

    # --- Language Model ---
    print(f'\nInitializing model: {API_TYPE} (uses your Claude Code subscription)...')
    model = language_model_utils.language_model_setup(
        api_type=API_TYPE,
        model_name=MODEL_NAME,
    )

    # --- Sentence Embedder ---
    print('Loading sentence embedder (first run downloads ~420MB)...')
    st_model = sentence_transformers.SentenceTransformer(
        'sentence-transformers/all-mpnet-base-v2'
    )
    embedder = lambda x: st_model.encode(x, show_progress_bar=False)

    # --- Test model ---
    print('Testing model connection...')
    test = model.sample_text('The settlement needs ')
    print(f'  Model says: "{test[:80]}..."')

    # --- Load prefabs ---
    print('\nLoading prefabs...')
    prefabs = {
        **helper_functions.get_package_classes(entity_prefabs),
        **helper_functions.get_package_classes(game_master_prefabs),
    }

    # --- Configure agents ---
    print('Configuring agents...')
    instances = []
    for agent in AGENTS:
        instances.append(prefab_lib.InstanceConfig(
            prefab='basic_with_plan__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': agent['name'],
                'goal': agent['goal'],
            },
        ))

    # --- Configure game masters ---
    # Initializer: generates backstories
    instances.append(prefab_lib.InstanceConfig(
        prefab='formative_memories_initializer__GameMaster',
        role=prefab_lib.Role.INITIALIZER,
        params={
            'name': 'initial setup rules',
            'next_game_master_name': 'settlement rules',
            'shared_memories': SHARED_MEMORIES,
            'player_specific_context': {
                agent['name']: agent['context'] for agent in AGENTS
            },
        },
    ))

    # Main GM: generic with rich world context
    instances.append(prefab_lib.InstanceConfig(
        prefab='generic__GameMaster',
        role=prefab_lib.Role.GAME_MASTER,
        params={
            'name': 'settlement rules',
            'acting_order': 'game_master_choice',
        },
    ))

    # --- Build config ---
    config = prefab_lib.Config(
        default_premise=PREMISE,
        default_max_steps=MAX_STEPS,
        prefabs=prefabs,
        instances=instances,
    )

    # --- Initialize simulation ---
    print('Initializing simulation (generating backstories — this takes a minute)...')
    sim = simulation.Simulation(
        config=config,
        model=model,
        embedder=embedder,
    )
    print('Simulation ready.\n')

    # --- Choose mode ---
    steps = args.steps
    if args.mode:
        choice = {'auto': '1', 'god': '2', 'both': '3'}[args.mode]
    else:
        print('How would you like to run the simulation?')
        print('  1. Auto-run (run all steps, then visualize)')
        print('  2. God Mode (interactive control)')
        print('  3. Both (auto-run first, then god mode)')
        choice = input('\nChoice [1/2/3]: ').strip()

    raw_log = []
    if choice == '2':
        raw_log, _ = god_mode(sim)
    elif choice == '3':
        print(f'\nRunning {steps} automatic steps...\n')
        results = sim.play(max_steps=steps, raw_log=raw_log)
        print('\n--- Auto-run complete. Entering god mode. ---')
        god_mode_log, _ = god_mode(sim)
        raw_log.extend(god_mode_log)
    else:
        print(f'\nRunning {steps} steps...\n')
        results = sim.play(max_steps=steps, raw_log=raw_log)

    # --- Print narrative log ---
    print('\n' + '=' * 60)
    print('  SIMULATION LOG')
    print('=' * 60)
    for i, entry in enumerate(raw_log):
        name = entry.get('entity_name', '???')
        action = entry.get('action', '')
        obs = entry.get('observation', '')
        text = action or obs
        if text:
            print(f'\n[Step {i+1}] {name}:')
            print(textwrap.fill(str(text), width=76, initial_indent='  ',
                               subsequent_indent='  '))

    # --- Visualize ---
    print('\nGenerating visualizations...')
    visualize_simulation(results if 'results' in dir() else None, raw_log)

    # --- Save full log ---
    log_path = 'sim_output/full_log.json'
    os.makedirs('sim_output', exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(raw_log, f, indent=2, default=str)
    print(f'Full log saved to {log_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
