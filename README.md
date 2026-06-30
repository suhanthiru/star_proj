# Custom Constellation Finder & Rendering

**Active Dev, WIP**
Algorithmic engine designed to parse stellar databases, compute star locations and shapes, and render constellation configs.

## Proj. Structure 

*  `START.py`: Entry point to launch code
*  `ingestion.py`: Data pipeline from star catalogs
*  `alignment_solver.py`: Core math/geometric algorithms to align structures
*  `gnn_engine.py`: Graph Neural Network or graph-based relational structural processing
*  `renderer.py`: Visualizes the computed celestial projections
*  `observatory.py`: Manages observer-centric coordinate systems and positioning
*  `vectorizer.py`: Transforms raw coordinate matrices into optimized vector spaces


##  Prereqs & External Data

This project relies on external celestial datasets that are ignored by version control due to file size limits. 
To run this locally, you must acquire the following files and place them in the root directory:

1.  **`de421.bsp`**: NASA JPL binary ephemeris file (can be fetched via `jplephem` or downloaded directly from JPL).
2.  **`hygdata_v42.csv.gz`**: The HYG star database catalog dataset.


Roadmap & Upcoming Features

- [x] Data injestion + Parser for stellar data
- [x] Word Vectorization/Normalizer
- [ ] Optimize the 'alignment_solver' algorithm for high-density star regions
- [ ] Implement interactive UI rendering inside 'renderer.py'
- [ ] Fine-tune the graph structures within 'gnn_engine.py' for pattern matching


All through the night, your glorious eyes
Were gazing down in mine,
And, with a full heart's thankful sighs,
I blessed that watch divine.
  -Emily Brontë
