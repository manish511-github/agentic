"""Initial migration for website_data and reddit_posts tables

Revision ID: abc123456789
Revises: 
Create Date: 2025-05-12 12:00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'abc123456789'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create website_data table
    op.create_table('website_data',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('url', sa.String(), nullable=False),
        sa.Column('title', sa.String()),
        sa.Column('description', sa.String()),
        sa.Column('target_audience', sa.String()),
        sa.Column('keywords', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('products_services', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_website_data_id'), 'website_data', ['id'], unique=False)

    # Create reddit_posts table
    op.create_table('reddit_posts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('agent_name', sa.String(), nullable=False),
        sa.Column('goals', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('instructions', sa.String()),
        sa.Column('subreddit', sa.String()),
        sa.Column('post_id', sa.String()),
        sa.Column('post_title', sa.String()),
        sa.Column('post_body', sa.String()),
        sa.Column('post_url', sa.String()),
        sa.Column('relevance_score', sa.Float()),
        sa.Column('sentiment_score', sa.Float()),
        sa.Column('comment_draft', sa.String()),
        sa.Column('status', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_reddit_posts_id'), 'reddit_posts', ['id'], unique=False)

def downgrade():
    op.drop_index(op.f('ix_reddit_posts_id'), table_name='reddit_posts')
    op.drop_table('reddit_posts')
    op.drop_index(op.f('ix_website_data_id'), table_name='website_data')
    op.drop_table('website_data')