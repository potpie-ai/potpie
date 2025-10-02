import logging
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from app.modules.parsing.models.inference_cache_model import InferenceCache
from app.modules.parsing.utils.content_hash import generate_content_hash

logger = logging.getLogger(__name__)


def analyze_cache_misses(
    nodes: List[Dict[str, Any]],
    db: Session
) -> Dict[str, Any]:
    """
    Analyze a set of nodes to understand why cache misses occur.

    Args:
        nodes: List of node dictionaries with 'node_id', 'text', 'node_type'
        db: Database session

    Returns:
        Dictionary with diagnostic information
    """
    diagnostics = {
        'total_nodes': 0,
        'missing_node_type': 0,
        'unresolved_references': 0,
        'cache_hits': 0,
        'cache_misses': 0,
        'examples': {
            'missing_type': [],
            'unresolved_ref': [],
            'cache_miss': []
        }
    }

    for node in nodes:
        # Skip nodes with no data
        if not node:
            continue

        node_id = node.get('node_id', 'unknown') if isinstance(node, dict) else 'unknown'
        node_text = node.get('text', '') if isinstance(node, dict) else ''
        node_type = node.get('node_type') if isinstance(node, dict) else None

        # Skip nodes with no text
        if not node_text:
            continue
        
        diagnostics['total_nodes'] += 1

        # Check for missing node_type
        if not node_type:
            diagnostics['missing_node_type'] += 1
            if len(diagnostics['examples']['missing_type']) < 5:
                diagnostics['examples']['missing_type'].append({
                    'node_id': str(node_id)[:16] if node_id else 'unknown',
                    'text_preview': str(node_text)[:50] if node_text else ''
                })

        # Check for unresolved references
        if node_text and 'Code replaced for brevity' in str(node_text):
            diagnostics['unresolved_references'] += 1
            if len(diagnostics['examples']['unresolved_ref']) < 5:
                diagnostics['examples']['unresolved_ref'].append({
                    'node_id': str(node_id)[:16] if node_id else 'unknown',
                    'text_preview': str(node_text)[:100] if node_text else ''
                })

        # Check if content would be cached
        try:
            content_hash = generate_content_hash(node_text, node_type)
            cache_entry = db.query(InferenceCache).filter(
                InferenceCache.content_hash == content_hash
            ).first()

            if cache_entry:
                diagnostics['cache_hits'] += 1
            else:
                diagnostics['cache_misses'] += 1
                if len(diagnostics['examples']['cache_miss']) < 5:
                    diagnostics['examples']['cache_miss'].append({
                        'node_id': str(node_id)[:16] if node_id else 'unknown',
                        'has_type': bool(node_type),
                        'hash': content_hash[:12] if content_hash else 'none',
                        'text_preview': str(node_text)[:50] if node_text else ''
                    })
        except Exception as e:
            logger.error(f"Error analyzing node {node_id}: {e}")

    # Calculate percentages
    if diagnostics['total_nodes'] > 0:
        diagnostics['missing_type_pct'] = (
            diagnostics['missing_node_type'] / diagnostics['total_nodes'] * 100
        )
        diagnostics['unresolved_ref_pct'] = (
            diagnostics['unresolved_references'] / diagnostics['total_nodes'] * 100
        )
        diagnostics['cache_hit_rate'] = (
            diagnostics['cache_hits'] / diagnostics['total_nodes'] * 100
        )

    return diagnostics


def log_diagnostics_summary(diagnostics: Dict[str, Any]) -> None:
    """Log a human-readable summary of diagnostics"""
    logger.info("=" * 60)
    logger.info("CACHE DIAGNOSTICS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total nodes analyzed: {diagnostics['total_nodes']}")
    logger.info(f"Cache hit rate: {diagnostics.get('cache_hit_rate', 0):.1f}%")
    logger.info(f"")
    logger.info(f"Issues detected:")
    logger.info(f"  - Missing node_type: {diagnostics['missing_node_type']} "
                f"({diagnostics.get('missing_type_pct', 0):.1f}%)")
    logger.info(f"  - Unresolved references: {diagnostics['unresolved_references']} "
                f"({diagnostics.get('unresolved_ref_pct', 0):.1f}%)")
    logger.info(f"")

    # Log examples
    if diagnostics['examples']['missing_type']:
        logger.info("Example nodes missing type:")
        for ex in diagnostics['examples']['missing_type']:
            logger.info(f"  - {ex['node_id']}: {ex['text_preview']}")

    if diagnostics['examples']['unresolved_ref']:
        logger.info("Example nodes with unresolved references:")
        for ex in diagnostics['examples']['unresolved_ref']:
            logger.info(f"  - {ex['node_id']}: {ex['text_preview']}")

    if diagnostics['examples']['cache_miss']:
        logger.info("Example cache misses:")
        for ex in diagnostics['examples']['cache_miss']:
            logger.info(f"  - {ex['node_id']} (type={ex['has_type']}): {ex['text_preview']}")

    logger.info("=" * 60)