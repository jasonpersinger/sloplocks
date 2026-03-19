# Design Spec: Data-Rich Matchup Cards

## Status: Approved
## Date: 2026-03-19

## Goals
1. **Increase Data Density:** Add Win Probability and Edge percentage back to the matchup cards.
2. **Head-to-Head Comparison:** Show stats for both the home and away teams to provide full model context.
3. **Inline Integration:** Maintain the minimalist aesthetic by placing stats inline with team names rather than adding new rows or bars.

## 1. UI Component: The Team Row
Each team row (Home and Away) within a match card will now contain three distinct horizontal zones:
- **Left (Flex-grow):** The Team Name (Bold `Oswald` text).
- **Center (Fixed Width):** The "Data Cluster".
    - **Win Probability:** Rounded to one decimal (e.g., `62.4%`).
    - **Edge:** Displayed in parentheses with a +/- sign (e.g., `(+8.2%)`).
- **Right (Fixed Width):** The "WIN" badge (visible only if that team is the model's current pick).

## 2. Visual Styling
- **Win Probability Text:** Standard white (`--text`).
- **Edge Text (Positive):** Slime green (`--slime`).
- **Edge Text (Negative):** Muted grey (`--text-muted`).
- **Alignment:** The Data Cluster will be right-aligned within its container to ensure that percentages line up vertically between the Home and Away rows.
- **Mobile Responsiveness:** To handle long team names on small screens, the team name will use `overflow: hidden; text-overflow: ellipsis; white-space: nowrap;` with a minimum flex-basis to ensure the data cluster and WIN badge remain fully visible.
- **Typography:** Stats will use `JetBrains Mono` for a precise, "data-heavy" feel.

## 3. Implementation Details
- **CSS:** Update `.team-row` to use `display: flex` with `justify-content: space-between`. Add a specific class for the data container (e.g., `.team-stats`) to handle alignment.
- **JS:** 
    - Modify `createMatchCard` to calculate or retrieve the edge for *both* sides (Home and Away).
    - If `match.edges` is present, use those values.
    - If only `match.model_probs` is present, display probabilities only or calculate a "virtual edge" if implied odds are available.
    - Update the template literal/DOM creation logic to inject these new spans.

## 4. Success Criteria
- Users can instantly see the win probability and edge for both sides of every matchup.
- The layout remains clean and doesn't feel cluttered or "tall".
- The "Slimegrinder" and "Slop Locks" sections both benefit from this increased transparency.
