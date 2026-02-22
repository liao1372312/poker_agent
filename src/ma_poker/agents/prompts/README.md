# Prompt Templates

This directory contains the 5 prompt templates used by OursAgent for RL-based prompt selection.

## Files

1. **balanced_gto.md** - Balanced / GTO Baseline strategy
2. **loose_passive.md** - Loose-Passive Exploit strategy (calling station)
3. **loose_aggressive.md** - Loose-Aggressive (LAG) strategy
4. **tight_passive.md** - Tight-Passive (Nit) strategy
5. **tight_aggressive.md** - Tight-Aggressive / Reg strategy

## Format

Each prompt template is a Markdown file that uses Python string formatting placeholders:

- `{hand}` - Player's hand cards
- `{public_cards}` - Public/community cards
- `{my_chips}` - Player's current chip count
- `{all_chips}` - All players' chip counts
- `{legal_actions}` - Available legal actions

## Editing

You can edit these files directly to modify the prompts. Changes will be automatically loaded by OursAgent on the next decision.

## Example

```markdown
You are an expert Texas Hold'em poker player.

Current situation:
- Your hand: {hand}
- Public cards: {public_cards}
- Your chips: {my_chips}
- All chips: {all_chips}
- Legal actions: {legal_actions}

Make a strategic decision. Respond with the action name.
```

## Notes

- Files are read using UTF-8 encoding
- If a file is missing or unreadable, OursAgent will fallback to a default template
- Changes to prompts take effect immediately (no restart required)

