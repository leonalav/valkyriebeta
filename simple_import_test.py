#!/usr/bin/env python
"""
Very simple test that just imports train_aio to check for import errors.
"""

import sys
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_import():
    """Test importing train_aio.py"""
    try:
        logger.info("Attempting to import train_aio.py...")
        import train_aio
        logger.info("✅ Successfully imported train_aio module!")
        return True
    except Exception as e:
        logger.error(f"❌ Error importing train_aio: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_import()
    if success:
        print("\n✅ IMPORT TEST PASSED: train_aio.py can be imported successfully!")
        sys.exit(0)
    else:
        print("\n❌ IMPORT TEST FAILED: train_aio.py could not be imported.")
        sys.exit(1) 