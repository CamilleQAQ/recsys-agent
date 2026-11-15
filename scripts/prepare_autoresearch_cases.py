"""Build real, frozen validation cases without training or index rebuilding."""
import argparse
from pathlib import Path
from recsys.platform.research_cases import recommendation_case, search_case, ads_case
from recsys.platform.autoresearch import write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('artifacts/research_cases'))
parser.add_argument('--businesses', default='recommendation,search,ads')
parser.add_argument('--rec-channels', type=Path)
parser.add_argument('--rec-splits', type=Path)
parser.add_argument('--rec-models', default='two_tower,itemcf,popularity')
parser.add_argument('--rec-users', type=int, default=32)
parser.add_argument('--search-config', default='configs/experiments/search_agent.toml')
parser.add_argument('--search-sessions', type=int, default=8)
parser.add_argument('--ads-dataset', default='kuasar_small')
parser.add_argument('--ads-models', default='din')
parser.add_argument('--ads-requests', type=int, default=32)
parser.add_argument('--ads-min-candidates', type=int, default=2)
args = parser.parse_args()
root = args.output.resolve()
manifest = {}
for business in args.businesses.split(','):
    output = root / f'{business}.json'
    if business == 'recommendation':
        if not args.rec_channels or not args.rec_splits:
            parser.error('recommendation requires explicit --rec-channels and --rec-splits')
        case = recommendation_case(args.rec_channels, args.rec_splits, output,
            models=args.rec_models.split(','), users=args.rec_users)
    elif business == 'search':
        case = search_case(output, config_path=args.search_config, sessions=args.search_sessions)
    elif business == 'ads':
        case = ads_case(output, dataset=args.ads_dataset, models=args.ads_models.split(','),
                        requests=args.ads_requests, min_candidates=args.ads_min_candidates)
    else:
        parser.error(f'Unknown business: {business}')
    manifest[business] = str(output)
    print(business, len(case['groups']), 'groups;', sum(len(g['positives']) for g in case['groups']), 'positives', flush=True)
write_json(root / 'manifest.json', manifest)
print(root / 'manifest.json')
