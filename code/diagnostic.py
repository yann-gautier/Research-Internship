import sktime

print("=== Dictionary-based classifiers ===")
try:
    import sktime.classification.dictionary_based as dict_based
    print("Available in dictionary_based:", dir(dict_based))
except ImportError as e:
    print(f"dictionary_based module issue: {e}")

print("\n=== All classification modules ===")
try:
    import sktime.classification as clf
    print("Available classification modules:", dir(clf))
except ImportError as e:
    print(f"classification module issue: {e}")

print("\n=== Searching for ElasticEnsemble alternatives ===")
# Try common locations for ElasticEnsemble
locations_to_try = [
    "sktime.classification.dictionary_based._elastic_ensemble",
    "sktime.classification.dictionary_based.elastic_ensemble", 
    "sktime.classification.ensemble._elastic_ensemble",
    "sktime.classification.ensemble.elastic_ensemble",
    "sktime.classification.hybrid._elastic_ensemble"
]

for location in locations_to_try:
    try:
        module_parts = location.split('.')
        module_path = '.'.join(module_parts[:-1])
        class_name = module_parts[-1]
        
        exec(f"from {module_path} import {class_name}")
        print(f"Found ElasticEnsemble-related class in: {location}")
    except ImportError:
        try:
            # Try importing ElasticEnsemble specifically
            exec(f"from {location} import ElasticEnsemble")
            print(f"Found ElasticEnsemble in: {location}")
        except ImportError:
            continue

print("\n=== Alternative ensemble classifiers ===")
try:
    from sktime.classification.ensemble import ComposableTimeSeriesForestClassifier
    print("Found: ComposableTimeSeriesForestClassifier")
except ImportError:
    pass

try:
    from sktime.classification.ensemble import TimeSeriesForestClassifier  
    print("Found: TimeSeriesForestClassifier")
except ImportError:
    pass

try:
    from sktime.classification.hybrid import HIVECOTEV2
    print("Found: HIVECOTEV2")
except ImportError:
    pass