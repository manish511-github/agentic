"""init

Revision ID: 9994af2a3ca2
Revises: 
Create Date: 2025-06-29 15:38:59.704654

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9994af2a3ca2'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('reddit_posts',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('subreddit', sa.String(), nullable=True),
    sa.Column('post_id', sa.String(), nullable=True),
    sa.Column('post_title', sa.String(), nullable=True),
    sa.Column('post_body', sa.String(), nullable=True),
    sa.Column('post_url', sa.String(), nullable=True),
    sa.Column('created_utc', sa.DateTime(), nullable=True),
    sa.Column('upvotes', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('post_id')
    )
    op.create_index('idx_post_id', 'reddit_posts', ['post_id'], unique=True)
    op.create_index(op.f('ix_reddit_posts_id'), 'reddit_posts', ['id'], unique=False)
    op.create_table('twitter_posts',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('agent_name', sa.String(), nullable=True),
    sa.Column('goals', sa.ARRAY(sa.String()), nullable=True),
    sa.Column('instructions', sa.String(), nullable=True),
    sa.Column('tweet_id', sa.String(), nullable=True),
    sa.Column('text', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('user_name', sa.String(), nullable=True),
    sa.Column('user_screen_name', sa.String(), nullable=True),
    sa.Column('retweet_count', sa.Integer(), nullable=True),
    sa.Column('favorite_count', sa.Integer(), nullable=True),
    sa.Column('relevance_score', sa.Float(), nullable=True),
    sa.Column('hashtags', sa.ARRAY(sa.String()), nullable=True),
    sa.Column('created', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_twitter_posts_id'), 'twitter_posts', ['id'], unique=False)
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=150), nullable=True),
    sa.Column('email', sa.String(length=255), nullable=True),
    sa.Column('hashed_password', sa.String(length=100), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('verified_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_username_email', 'users', ['username', 'email'], unique=False)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=False)
    op.create_table('website_data',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('url', sa.String(), nullable=False),
    sa.Column('title', sa.String(), nullable=True),
    sa.Column('description', sa.String(), nullable=True),
    sa.Column('target_audience', sa.String(), nullable=True),
    sa.Column('keywords', sa.JSON(), nullable=True),
    sa.Column('products_services', sa.JSON(), nullable=True),
    sa.Column('main_category', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_website_data_id'), 'website_data', ['id'], unique=False)
    op.create_table('projects',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('uuid', sa.String(), nullable=True),
    sa.Column('title', sa.String(), nullable=False),
    sa.Column('description', sa.String(), nullable=True),
    sa.Column('target_audience', sa.String(), nullable=True),
    sa.Column('website_url', sa.String(), nullable=True),
    sa.Column('category', sa.String(), nullable=True),
    sa.Column('priority', sa.String(), nullable=True),
    sa.Column('due_date', sa.DateTime(), nullable=True),
    sa.Column('budget', sa.String(), nullable=True),
    sa.Column('team', sa.JSON(), nullable=True),
    sa.Column('tags', sa.String(), nullable=True),
    sa.Column('competitors', sa.JSON(), nullable=True),
    sa.Column('keywords', sa.JSON(), nullable=True),
    sa.Column('excluded_keywords', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('owner_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_projects_id'), 'projects', ['id'], unique=False)
    op.create_index(op.f('ix_projects_uuid'), 'projects', ['uuid'], unique=True)
    op.create_table('user_tokens',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('access_key', sa.String(length=250), nullable=True),
    sa.Column('refresh_key', sa.String(length=250), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('expires_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_user_tokens_access_key'), 'user_tokens', ['access_key'], unique=False)
    op.create_index(op.f('ix_user_tokens_refresh_key'), 'user_tokens', ['refresh_key'], unique=False)
    op.create_table('agents',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('agent_name', sa.String(), nullable=False),
    sa.Column('description', sa.String(), nullable=True),
    sa.Column('agent_platform', sa.String(), nullable=False),
    sa.Column('agent_status', sa.String(), nullable=True),
    sa.Column('goals', sa.String(), nullable=True),
    sa.Column('instructions', sa.String(), nullable=True),
    sa.Column('expectations', sa.String(), nullable=True),
    sa.Column('keywords', sa.ARRAY(sa.String()), nullable=True),
    sa.Column('project_id', sa.String(), nullable=True),
    sa.Column('mode', sa.String(), nullable=True),
    sa.Column('review_minutes', sa.Integer(), nullable=True),
    sa.Column('advanced_settings', sa.JSON(), nullable=True),
    sa.Column('platform_settings', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('last_run', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['project_id'], ['projects.uuid'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_agents_id'), 'agents', ['id'], unique=False)
    op.create_table('agent_results',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('agent_id', sa.Integer(), nullable=True),
    sa.Column('project_id', sa.String(), nullable=True),
    sa.Column('results', sa.JSON(), nullable=True),
    sa.Column('status', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ),
    sa.ForeignKeyConstraint(['project_id'], ['projects.uuid'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_agent_results_id'), 'agent_results', ['id'], unique=False)
    op.create_table('schedules',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('agent_id', sa.Integer(), nullable=True),
    sa.Column('schedule_type', sa.Enum('daily', 'weekly', 'monthly', name='scheduletypeenum'), nullable=False),
    sa.Column('schedule_time', sa.DateTime(), nullable=True),
    sa.Column('days_of_week', sa.ARRAY(sa.String()), nullable=True),
    sa.Column('day_of_month', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.CheckConstraint("schedule_type != 'monthly' OR day_of_month IS NOT NULL", name='chk_day_of_month'),
    sa.CheckConstraint("schedule_type != 'weekly' OR days_of_week IS NOT NULL", name='chk_days_of_week'),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('agent_id', name='uix_agent_schedule')
    )
    op.create_index(op.f('ix_schedules_id'), 'schedules', ['id'], unique=False)
    op.create_table('executions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('schedule_id', sa.Integer(), nullable=True),
    sa.Column('agent_id', sa.Integer(), nullable=True),
    sa.Column('schedule_time', sa.DateTime(), nullable=False),
    sa.Column('status', sa.Enum('scheduled', 'queued', 'running', 'completed', 'failed', name='executionstatusenum'), nullable=False),
    sa.Column('results', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ),
    sa.ForeignKeyConstraint(['schedule_id'], ['schedules.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_executions_id'), 'executions', ['id'], unique=False)
    op.create_table('reddit_agent_execution_mapper',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('execution_id', sa.Integer(), nullable=True),
    sa.Column('agent_id', sa.Integer(), nullable=True),
    sa.Column('post_id', sa.String(), nullable=True),
    sa.Column('relevance_score', sa.Float(), nullable=True),
    sa.Column('comment_draft', sa.String(), nullable=True),
    sa.Column('status', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ),
    sa.ForeignKeyConstraint(['execution_id'], ['executions.id'], ),
    sa.ForeignKeyConstraint(['post_id'], ['reddit_posts.post_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_execution_agent', 'reddit_agent_execution_mapper', ['execution_id', 'agent_id'], unique=False)
    op.create_index(op.f('ix_reddit_agent_execution_mapper_id'), 'reddit_agent_execution_mapper', ['id'], unique=False)
    # ### end Alembic commands ###

def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_reddit_agent_execution_mapper_id'), table_name='reddit_agent_execution_mapper')
    op.drop_index('idx_execution_agent', table_name='reddit_agent_execution_mapper')
    op.drop_table('reddit_agent_execution_mapper')
    op.drop_index(op.f('ix_executions_id'), table_name='executions')
    op.drop_table('executions')
    op.drop_index(op.f('ix_schedules_id'), table_name='schedules')
    op.drop_table('schedules')
    op.drop_index(op.f('ix_agent_results_id'), table_name='agent_results')
    op.drop_table('agent_results')
    op.drop_index(op.f('ix_agents_id'), table_name='agents')
    op.drop_table('agents')
    op.drop_index(op.f('ix_user_tokens_refresh_key'), table_name='user_tokens')
    op.drop_index(op.f('ix_user_tokens_access_key'), table_name='user_tokens')
    op.drop_table('user_tokens')
    op.drop_index(op.f('ix_projects_uuid'), table_name='projects')
    op.drop_index(op.f('ix_projects_id'), table_name='projects')
    op.drop_table('projects')
    op.drop_index(op.f('ix_website_data_id'), table_name='website_data')
    op.drop_table('website_data')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_index('idx_username_email', table_name='users')
    op.drop_table('users')
    op.drop_index(op.f('ix_twitter_posts_id'), table_name='twitter_posts')
    op.drop_table('twitter_posts')
    op.drop_index(op.f('ix_reddit_posts_id'), table_name='reddit_posts')
    op.drop_index('idx_post_id', table_name='reddit_posts')
    op.drop_table('reddit_posts')
    # ### end Alembic commands ###