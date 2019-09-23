import os

DEV_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '%(asctime)s %(levelname)s - %(message)s'
        },
        'with_filename': {
            'format': '%(asctime)s (%(hostname)s-%(pathname)s:%(lineno)d '
                      '%(levelname)s [%(current_execution)s] - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        }
    },
    'loggers': {
        'my_module': {
            'level': 'INFO',
            'handlers': ['console'],
            'propagate': 'no'
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console'],
        'propagate': 'no'
    }
}


PROD_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '%(asctime)s %(levelname)s - %(message)s'
        },
        'with_filename': {
            'format': '%(asctime)s (%(hostname)s-%(pathname)s:%(lineno)d '
                      '%(levelname)s [%(current_execution)s] - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        }
    },
    'loggers': {
        'my_module': {
            'level': 'INFO',
            'handlers': ['console'],
            'propagate': 'no'
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console'],
        'propagate': 'no'
    }
}