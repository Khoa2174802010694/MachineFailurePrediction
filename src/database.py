
import sqlite3
from pathlib import Path


# =====================================================
# 1. CẤU HÌNH DATABASE
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "database" / "machine_management.db"


def get_connection():
    """
    Tạo kết nối SQLite.
    Tự động tạo thư mục database nếu chưa tồn tại.
    """
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


# =====================================================
# 2. KHỞI TẠO CÁC BẢNG
# =====================================================

def init_database():
    """
    Khởi tạo các bảng cần thiết cho hệ thống.
    Chỉ tạo bảng nếu chúng chưa tồn tại.
    """

    with get_connection() as connection:

        # -------------------------------------------------
        # BẢNG 1: THIẾT BỊ
        # -------------------------------------------------

        connection.execute("""
            CREATE TABLE IF NOT EXISTS machines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                machine_code TEXT NOT NULL UNIQUE,
                machine_name TEXT NOT NULL,

                machine_type TEXT NOT NULL
                    CHECK (machine_type IN ('L', 'M', 'H')),

                location TEXT,
                manufacturer TEXT,
                model_number TEXT,

                installation_date TEXT,

                status TEXT NOT NULL DEFAULT 'active'
                    CHECK (
                        status IN (
                            'active',
                            'maintenance',
                            'inactive'
                        )
                    ),

                notes TEXT,

                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # -------------------------------------------------
        # BẢNG 2: LỊCH SỬ DỰ ĐOÁN
        # -------------------------------------------------

        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                machine_id INTEGER,

                source TEXT NOT NULL DEFAULT 'manual'
                    CHECK (
                        source IN ('manual', 'csv', 'monitoring')
                    ),

                air_temperature REAL NOT NULL,
                process_temperature REAL NOT NULL,
                rotational_speed REAL NOT NULL,
                torque REAL NOT NULL,
                tool_wear REAL NOT NULL,

                failure_probability REAL NOT NULL,
                predicted_failure INTEGER NOT NULL
                    CHECK (predicted_failure IN (0, 1)),

                threshold REAL NOT NULL,
                model_name TEXT NOT NULL DEFAULT 'Random Forest',

                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (machine_id)
                    REFERENCES machines(id)
                    ON DELETE SET NULL
            )
        """)

        # -------------------------------------------------
        # BẢNG 3: PHIẾU BẢO TRÌ
        # -------------------------------------------------

        connection.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                machine_id INTEGER NOT NULL,

                title TEXT NOT NULL,
                description TEXT,

                priority TEXT NOT NULL DEFAULT 'medium'
                    CHECK (
                        priority IN ('low', 'medium', 'high', 'critical')
                    ),

                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (
                        status IN (
                            'pending',
                            'in_progress',
                            'completed',
                            'cancelled'
                        )
                    ),

                scheduled_date TEXT,
                completed_at TEXT,

                result_notes TEXT,

                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (machine_id)
                    REFERENCES machines(id)
                    ON DELETE CASCADE
            )
        """)

        # -------------------------------------------------
        # 3. INDEX TĂNG TỐC TRUY VẤN
        # -------------------------------------------------

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_machine
            ON prediction_history(machine_id)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_created
            ON prediction_history(created_at)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_maintenance_machine
            ON maintenance_records(machine_id)
        """)

        connection.commit()


# =====================================================
# 4. KIỂM TRA DATABASE
# =====================================================

def check_database():
    """
    Trả về danh sách các bảng hiện có trong database.
    Dùng để kiểm tra quá trình khởi tạo.
    """

    with get_connection() as connection:
        rows = connection.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
        """).fetchall()

    return [row["name"] for row in rows]


# =====================================================
# 5. QUẢN LÝ THIẾT BỊ (CRUD)
# =====================================================

def add_machine(
    machine_code,
    machine_name,
    machine_type,
    location=None,
    manufacturer=None,
    model_number=None,
    installation_date=None,
    notes=None
):
    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO machines (
                machine_code,
                machine_name,
                machine_type,
                location,
                manufacturer,
                model_number,
                installation_date,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            machine_code.strip(),
            machine_name.strip(),
            machine_type,
            location,
            manufacturer,
            model_number,
            installation_date,
            notes
        ))

        return cursor.lastrowid


def get_all_machines(include_inactive=True):
    with get_connection() as connection:
        query = "SELECT * FROM machines"

        if not include_inactive:
            query += " WHERE status != 'inactive'"

        query += " ORDER BY created_at DESC"

        return connection.execute(query).fetchall()


def get_machine_by_id(machine_id):
    with get_connection() as connection:
        return connection.execute("""
            SELECT *
            FROM machines
            WHERE id = ?
        """, (machine_id,)).fetchone()


def update_machine(machine_id, **kwargs):
    allowed_fields = {
        "machine_code",
        "machine_name",
        "machine_type",
        "location",
        "manufacturer",
        "model_number",
        "installation_date",
        "status",
        "notes"
    }

    fields = {
        key: value
        for key, value in kwargs.items()
        if key in allowed_fields
    }

    if not fields:
        return False

    fields["updated_at"] = "CURRENT_TIMESTAMP"

    set_clause = ", ".join(
        f"{key} = ?" if value != "CURRENT_TIMESTAMP"
        else f"{key} = CURRENT_TIMESTAMP"
        for key, value in fields.items()
    )

    values = [
        value for value in fields.values()
        if value != "CURRENT_TIMESTAMP"
    ]

    values.append(machine_id)

    with get_connection() as connection:
        cursor = connection.execute(
            f"UPDATE machines SET {set_clause} WHERE id = ?",
            values
        )

        return cursor.rowcount > 0


def delete_machine(machine_id):
    """
    Ngừng theo dõi thiết bị thay vì xóa dữ liệu vĩnh viễn.
    Lịch sử dự đoán và bảo trì vẫn được giữ lại.
    """
    return update_machine(machine_id, status="inactive")


# =====================================================
# 6. LƯU VÀ TRUY VẤN LỊCH SỬ DỰ ĐOÁN
# =====================================================

def save_prediction(
    machine_id,
    input_data,
    probability,
    prediction,
    threshold,
    source="manual",
    model_name="Random Forest"
):
    """
    Lưu một lần dự đoán vào lịch sử.

    machine_id:
        ID thiết bị trong bảng machines.
        Có thể là None nếu dự đoán thử nghiệm.

    input_data:
        Dictionary chứa các thông số đầu vào.
    """

    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO prediction_history (
                machine_id,
                source,
                air_temperature,
                process_temperature,
                rotational_speed,
                torque,
                tool_wear,
                failure_probability,
                predicted_failure,
                threshold,
                model_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            machine_id,
            source,
            float(input_data["Air temperature [K]"]),
            float(input_data["Process temperature [K]"]),
            float(input_data["Rotational speed [rpm]"]),
            float(input_data["Torque [Nm]"]),
            float(input_data["Tool wear [min]"]),
            float(probability),
            int(prediction),
            float(threshold),
            model_name
        ))

        return cursor.lastrowid


def get_prediction_history(machine_id=None, limit=100):
    """
    Lấy lịch sử dự đoán.
    Nếu machine_id=None, lấy lịch sử của toàn hệ thống.
    """

    with get_connection() as connection:

        if machine_id is None:
            rows = connection.execute("""
                SELECT
                    p.*,
                    m.machine_code,
                    m.machine_name
                FROM prediction_history p
                LEFT JOIN machines m
                    ON p.machine_id = m.id
                ORDER BY p.created_at DESC, p.id DESC
                LIMIT ?
            """, (limit,)).fetchall()

        else:
            rows = connection.execute("""
                SELECT
                    p.*,
                    m.machine_code,
                    m.machine_name
                FROM prediction_history p
                LEFT JOIN machines m
                    ON p.machine_id = m.id
                WHERE p.machine_id = ?
                ORDER BY p.created_at DESC, p.id DESC
                LIMIT ?
            """, (machine_id, limit)).fetchall()

        return rows


# =====================================================
# 7. QUẢN LÝ PHIẾU BẢO TRÌ
# =====================================================
def add_maintenance_record(
    machine_id,
    title,
    description=None,
    priority="medium",
    scheduled_date=None
):
    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO maintenance_records (
                machine_id,
                title,
                description,
                priority,
                scheduled_date
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            machine_id,
            title.strip(),
            description,
            priority,
            scheduled_date
        ))

        return cursor.lastrowid


def get_maintenance_records(
    machine_id=None,
    status=None,
    limit=500
):
    with get_connection() as connection:

        query = """
            SELECT
                r.*,
                m.machine_code,
                m.machine_name
            FROM maintenance_records r
            JOIN machines m
                ON r.machine_id = m.id
            WHERE 1 = 1
        """

        params = []

        if machine_id is not None:
            query += " AND r.machine_id = ?"
            params.append(machine_id)

        if status is not None:
            query += " AND r.status = ?"
            params.append(status)

        query += """
            ORDER BY
                CASE r.priority
                    WHEN 'critical' THEN 1
                    WHEN 'high' THEN 2
                    WHEN 'medium' THEN 3
                    ELSE 4
                END,
                r.created_at DESC
            LIMIT ?
        """

        params.append(limit)

        return connection.execute(
            query,
            params
        ).fetchall()


def update_maintenance_record(
    record_id,
    status=None,
    priority=None,
    scheduled_date=None,
    description=None,
    result_notes=None
):
    allowed_statuses = {
        "pending",
        "in_progress",
        "completed",
        "cancelled"
    }

    allowed_priorities = {
        "low",
        "medium",
        "high",
        "critical"
    }

    fields = {}
    values = []

    if status is not None:
        if status not in allowed_statuses:
            raise ValueError("Trạng thái bảo trì không hợp lệ.")

        fields["status"] = status

    if priority is not None:
        if priority not in allowed_priorities:
            raise ValueError("Mức ưu tiên không hợp lệ.")

        fields["priority"] = priority

    if scheduled_date is not None:
        fields["scheduled_date"] = scheduled_date

    if description is not None:
        fields["description"] = description

    if result_notes is not None:
        fields["result_notes"] = result_notes

    if not fields:
        return False

    set_clause = ", ".join(
        f"{field} = ?" for field in fields
    )

    values = list(fields.values())

    if status == "completed":
        set_clause += ", completed_at = CURRENT_TIMESTAMP"
    elif status is not None:
        set_clause += ", completed_at = NULL"

    set_clause += ", updated_at = CURRENT_TIMESTAMP"
    values.append(record_id)

    with get_connection() as connection:
        cursor = connection.execute(
            f"""
            UPDATE maintenance_records
            SET {set_clause}
            WHERE id = ?
            """,
            values
        )

        return cursor.rowcount > 0
