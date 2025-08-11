"""版本信息管理"""

__version__ = "1.0.0-dev"

VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'status': 'dev'  # dev, alpha, beta, rc, stable
}

def get_version_info():
    """获取详细版本信息"""
    return VERSION_INFO

def get_version_string():
    """获取版本字符串"""
    return f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['patch']}"