from enum import Enum

PH_DIMS = (2960, 2960)

NTC = 'nontargeting'

VAL = [('20200202_6W-LaC024A', 'A1'),
       ('20200202_6W-LaC024E', 'A1'),
       ('20200206_6W-LaC025A', 'A1'),
       ('20200202_6W-LaC024D', 'A1'),
       ('20200202_6W-LaC024F', 'A1'),
       ('20200206_6W-LaC025B', 'A1'),
       ]

TEST = VAL

CHANNELS = ['A594', 'AF750', 'DAPI-GFP_0', 'DAPI-GFP_1']

class Column(Enum):
    index = 'index'
    plate = 'plate'
    well = 'well'
    tile = 'tile'
    sgRNA = 'sgRNA_0'
    gene = 'gene_symbol_0'
    cell_y = 'cell_i'
    cell_x = 'cell_j'
    cell_cycle_stage = 'stage'
    batch = 'batch'
    uid = 'UID'
    function = 'function'

class cell_cycle(Enum):
    mitotic = 'mitotic'
    interphase = 'interphase'
