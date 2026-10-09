-- ============================================================
-- MySQL 8.0 建库建用户
-- ------------------------------------------------------------
-- 执行前：把下面 <强口令> 替换为真实密码，且与 .env 的 DB_PASSWORD 完全一致
-- 执行：  mysql -u root -p < deploy/setup_mysql8.sql
-- ============================================================

-- 数据库：utf8mb4 字符集 + general_ci 排序
-- （8.0 默认是 utf8mb4_0900_ai_ci，这里显式 general_ci 以对齐 5.7 行为，避免排序差异）
CREATE DATABASE IF NOT EXISTS ml_db CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;

-- 用户：用 mysql_native_password 认证（兼容性最稳；PyMySQL 1.2 也支持 8.0 默认的 caching_sha2）
-- 若服务器装的是 MySQL 8.4+（默认禁用 mysql_native_password）报错，
-- 把下面这行改成：CREATE USER IF NOT EXISTS 'ml_user'@'%' IDENTIFIED BY '<强口令>';
CREATE USER IF NOT EXISTS 'ml_user'@'%' IDENTIFIED WITH mysql_native_password BY '<强口令>';
GRANT ALL PRIVILEGES ON ml_db.* TO 'ml_user'@'%';
FLUSH PRIVILEGES;

-- 时区表：Linux 一般已从 /usr/share/zoneinfo 自动加载。
-- 若 Django 报时区相关警告，执行（任选一行，root 权限）：
--   mysql_tzinfo_to_sql /usr/share/zoneinfo | mysql -u root -p mysql
